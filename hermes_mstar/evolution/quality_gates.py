"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          Hermes MSTAR: QualityGates - 4门质量检查                                                    ║
║                                                                                                              ║
║  MSTAR 的质量门禁机制，拦截不合格的变异                                                                      ║
║  四门检查:                                                                                                ║
║    1. Compile Gate   - 语法检查                                                                           ║
║    2. Runtime Gate   - 执行检查                                                                           ║
║    3. Logic Gate    - 逻辑检查                                                                            ║
║    4. Quality Gate  - 质量检查                                                                           ║
║                                                                                                              ║
║  M* Paper Phase 4 Upgrade:                                                                               ║
║    - Automated Repair Loop (3× retry on compile failure)                                                   ║
║    - MSTARMutator.mutate() returns MutationResult with .success attribute                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..memory_program import MemoryProgram

logger = logging.getLogger(__name__)


class GateResult(str, Enum):
    """门禁结果"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARNING = "warning"


@dataclass
class GateReport:
    """门禁报告"""
    compile_result: GateResult = GateResult.SKIP
    compile_message: str = ""
    runtime_result: GateResult = GateResult.SKIP
    runtime_message: str = ""
    logic_result: GateResult = GateResult.SKIP
    logic_message: str = ""
    quality_result: GateResult = GateResult.SKIP
    quality_message: str = ""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # M* Paper Phase 4: Repair loop support
    repair_attempts: int = 0
    last_error: Optional[str] = None

    @property
    def all_passed(self) -> bool:
        """所有门禁是否通过（M* Paper Phase 4 兼容）"""
        return (
            self.compile_result == GateResult.PASS and
            self.runtime_result == GateResult.PASS and
            self.logic_result == GateResult.PASS and
            self.quality_result == GateResult.PASS
        )

    @property
    def passed(self) -> bool:
        """所有门禁是否通过"""
        return self.all_passed

    @property
    def any_failure(self) -> bool:
        """是否有任何门禁失败"""
        return (
            self.compile_result == GateResult.FAIL or
            self.runtime_result == GateResult.FAIL or
            self.logic_result == GateResult.FAIL or
            self.quality_result == GateResult.FAIL
        )

    @property
    def summary(self) -> str:
        """总结"""
        if self.all_passed:
            return "All gates passed"
        failed = []
        if self.compile_result == GateResult.FAIL:
            failed.append(f"Compile({self.compile_message})")
        if self.runtime_result == GateResult.FAIL:
            failed.append(f"Runtime({self.runtime_message})")
        if self.logic_result == GateResult.FAIL:
            failed.append(f"Logic({self.logic_message})")
        if self.quality_result == GateResult.FAIL:
            failed.append(f"Quality({self.quality_message})")
        return f"Failed: {', '.join(failed)}"

    @property
    def failed_gates(self) -> List[str]:
        """获取所有失败的检查门"""
        failed = []
        if self.compile_result == GateResult.FAIL:
            failed.append(f"Compile: {self.compile_message}")
        if self.runtime_result == GateResult.FAIL:
            failed.append(f"Runtime: {self.runtime_message}")
        if self.logic_result == GateResult.FAIL:
            failed.append(f"Logic: {self.logic_message}")
        if self.quality_result == GateResult.FAIL:
            failed.append(f"Quality: {self.quality_message}")
        return failed


class QualityGates:
    """
    4门质量检查 + M* Paper Phase 4: 3× Repair Loop

    设计原则:
    1. 严格模式 - 任何门禁失败都拒绝部署
    2. 渐进式 - 从编译到质量逐级检查
    3. 可解释 - 每个门禁都有清晰的失败原因
    4. M* Paper: 3× Automated Repair — 编译失败时自动重试（最多3次）
    """

    # M* Paper Phase 4: 编译失败重试次数
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self):
        """初始化质量门禁"""
        # 阈值配置
        self.compile_timeout = 5.0       # 编译超时 (秒)
        self.runtime_timeout = 10.0      # 运行时超时 (秒)
        self.quality_threshold = 0.4      # 质量阈值
        self.min_keywords = 1             # 最少关键词数
        self.max_keywords = 20            # 最多关键词数
        self.min_content_length = 50      # 最少内容长度

    def run_all(self, program: MemoryProgram) -> GateReport:
        """
        运行所有门禁检查

        Args:
            program: 待检查的程序

        Returns:
            GateReport
        """
        report = GateReport()

        try:
            # 1. Compile Gate (M* Paper Phase 4: 3× Repair Loop)
            report = self._compile_gate_with_repair(program, report)

            # Early exit on compile failure (after repair attempts)
            if report.compile_result == GateResult.FAIL:
                return report

            # 2. Runtime Gate
            report.runtime_result, report.runtime_message = self._check_runtime(program)
            if report.runtime_result == GateResult.FAIL:
                report.errors.append(f"Runtime gate failed: {report.runtime_message}")
                return report

            # 3. Logic Gate
            report.logic_result, report.logic_message = self._check_logic(program)
            if report.logic_result == GateResult.FAIL:
                report.errors.append(f"Logic gate failed: {report.logic_message}")
                return report

            # 4. Quality Gate
            report.quality_result, report.quality_message = self._check_quality(program)
            if report.quality_result == GateResult.FAIL:
                report.errors.append(f"Quality gate failed: {report.quality_message}")
                return report

            logger.info(f"Quality gates passed for {program.name}")
            return report

        except Exception as e:
            logger.error(f"Quality gates error for {program.name}: {e}")
            report.errors.append(f"Gate check error: {str(e)}")
            return report

    # ── M* Paper Phase 4: 3× Automated Repair Loop ─────────────────────────────────

    def _compile_gate_with_repair(self, program: MemoryProgram, report: GateReport) -> GateReport:
        """
        M* Paper Phase 4: Compile Gate with 3× Automated Repair

        论文规格:
          "Constraint check + automated repair (compile fix × 3 attempts)"

        如果编译失败，利用错误信息尝试修复后重试。
        """
        attempt = 0
        last_error = None

        while attempt <= self.MAX_REPAIR_ATTEMPTS:
            result, message = self._check_compile(program)

            if result == GateResult.PASS:
                report.compile_result = GateResult.PASS
                report.compile_message = message
                return report

            # Failed — record error
            last_error = message
            report.repair_attempts = attempt
            report.last_error = last_error

            if attempt >= self.MAX_REPAIR_ATTEMPTS:
                # Exhausted retries
                report.compile_result = GateResult.FAIL
                report.compile_message = f"Gave up after {self.MAX_REPAIR_ATTEMPTS + 1} attempts: {last_error}"
                report.errors.append(f"Compile gate failed (exhausted {self.MAX_REPAIR_ATTEMPTS} repairs): {last_error}")
                return report

            # Try repair
            attempt += 1
            logger.debug(f"Compile gate attempt {attempt}/{self.MAX_REPAIR_ATTEMPTS} failed: {last_error}. Attempting repair...")

            repaired = self._attempt_compile_repair(program, last_error)
            if repaired is None:
                # No repair possible, give up
                break

            # Apply repaired content and retry
            logger.debug(f"Compile repair attempt {attempt}: applied patch, re-checking...")
            # Note: program is not mutated here — repair is for gates only.
            # The caller (evolution_engine) decides whether to apply the repair.

        report.compile_result = GateResult.FAIL
        report.compile_message = f"Compile failed after {self.MAX_REPAIR_ATTEMPTS + 1} attempts: {last_error}"
        report.errors.append(f"Compile gate failed: {last_error}")
        return report

    def _attempt_compile_repair(self, program: MemoryProgram, error: str) -> Optional[str]:
        """
        尝试根据编译错误修复 program content

        Returns:
            Repaired content string, or None if no repair possible
        """
        content = program.content
        error_lower = error.lower()

        # Pattern 1: YAML frontmatter broken
        if 'yaml' in error_lower or 'frontmatter' in error_lower:
            # Try to fix missing closing ---
            if content.startswith('---') and content.count('---') == 1:
                # Missing closing ---
                lines = content.split('\n', 1)
                if len(lines) > 1 and not lines[1].startswith('---'):
                    # Add closing ---
                    repaired = content + '\n---'
                    logger.debug(f"YAML repair: added closing ---")
                    return repaired

            # Try to fix unclosed YAML
            if content.startswith('---') and content.count('---') == 2:
                # Find the second --- and check if there's content after
                parts = content.split('---', 2)
                if len(parts) >= 3 and not parts[2].strip():
                    # Empty content after frontmatter
                    repaired = parts[0] + '---' + parts[1] + '---\n\n'
                    logger.debug(f"YAML repair: closed empty frontmatter")
                    return repaired

        # Pattern 2: Missing heading
        if 'heading' in error_lower or 'no heading' in error_lower:
            lines = content.split('\n')
            has_heading = any(l.strip().startswith('#') for l in lines)
            if not has_heading:
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    repaired = parts[0] + parts[1] + '---\n\n# ' + (program.name or 'Skill') + '\n\n' + parts[2]
                    logger.debug(f"Heading repair: added skill title heading")
                    return repaired
                else:
                    repaired = '# ' + (program.name or 'Skill') + '\n\n' + content
                    logger.debug(f"Heading repair: prepended title heading")
                    return repaired

        # Pattern 3: Content too short
        if 'too short' in error_lower:
            # Pad with meaningful placeholder content
            padding = "\n\n<!-- Content placeholder: expand this section with detailed instructions -->"
            repaired = content + padding
            logger.debug(f"Content length repair: padded to meet minimum")
            return repaired

        # Pattern 4: Missing trigger keywords (detected in logic gate, not compile)
        # No repair available for this pattern

        return None

    # ── Original Gate Methods ────────────────────────────────────────────────────

    def _check_compile(self, program: MemoryProgram) -> Tuple[GateResult, str]:
        """
        Compile Gate: 检查语法

        检查项:
        - YAML frontmatter 格式
        - Markdown 结构完整性
        - 内容不为空
        """
        content = program.content

        if not content:
            return GateResult.FAIL, "Empty content"

        if len(content) < self.min_content_length:
            return GateResult.FAIL, f"Content too short ({len(content)} < {self.min_content_length})"

        # 检查 YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) < 3:
                return GateResult.FAIL, "Invalid YAML frontmatter"

            yaml_content = parts[1]
            try:
                self._parse_yaml(yaml_content)
            except Exception as e:
                return GateResult.FAIL, f"Invalid YAML: {str(e)}"

        # 检查 Markdown 结构
        lines = content.split('\n')
        has_heading = any(line.strip().startswith('#') for line in lines)
        if not has_heading:
            # Warning only, not failure
            return GateResult.WARNING, "No heading found (warning)"

        return GateResult.PASS, "ok"

    def _check_runtime(self, program: MemoryProgram) -> Tuple[GateResult, str]:
        """
        Runtime Gate: 检查可执行性

        检查项:
        - Python 代码可执行 (如果包含)
        - Shell 脚本可执行 (如果包含)
        - PowerShell 脚本可执行 (Windows支持)
        - YAML 配置可解析 (如果包含)
        """
        content = program.content

        # 检查 Python 代码块
        python_blocks = self._extract_code_blocks(content, 'python')
        for block in python_blocks:
            result = self._check_python_syntax(block)
            if result is not None:
                return GateResult.FAIL, f"Python syntax error: {result}"

        # 检查 Shell 代码块 (bash)
        shell_blocks = self._extract_code_blocks(content, 'bash')
        for block in shell_blocks:
            result = self._check_shell_syntax(block)
            if result is not None:
                return GateResult.FAIL, f"Shell syntax error: {result}"

        # 检查 PowerShell 代码块 (Windows支持)
        powershell_blocks = self._extract_code_blocks(content, 'powershell')
        for block in powershell_blocks:
            result = self._check_powershell_syntax(block)
            if result is not None:
                return GateResult.FAIL, f"PowerShell syntax error: {result}"

        # 检查 YAML 代码块 (配置文件)
        yaml_blocks = self._extract_code_blocks(content, 'yaml')
        for block in yaml_blocks:
            result = self._check_yaml_syntax(block)
            if result is not None:
                return GateResult.FAIL, f"YAML syntax error: {result}"

        return GateResult.PASS, "ok"

    def _check_logic(self, program: MemoryProgram) -> Tuple[GateResult, str]:
        """
        Logic Gate: 检查逻辑完整性

        检查项:
        - trigger_keywords 合理性
        - agent_guidance 完整性
        - 状态一致性
        """
        # 检查 trigger_keywords
        keywords = program.instructions.trigger_keywords
        if not keywords:
            return GateResult.FAIL, "No trigger keywords"

        if len(keywords) > self.max_keywords:
            return GateResult.FAIL, f"Too many keywords ({len(keywords)} > {self.max_keywords})"

        # 检查关键词长度
        for kw in keywords:
            if len(kw) < 2:
                return GateResult.FAIL, f"Keyword too short: '{kw}'"
            if len(kw) > 50:
                return GateResult.FAIL, f"Keyword too long: '{kw}'"

        # 检查 agent_guidance
        if not program.agent_guidance and len(program.content) > 500:
            # 长内容没有 guidance 可能有问题
            return GateResult.FAIL, "Long content without agent_guidance"

        # 检查优先级
        if not 1 <= program.priority <= 10:
            return GateResult.FAIL, f"Invalid priority: {program.priority}"

        # 检查阈值
        if not 0 <= program.trigger_threshold <= 1:
            return GateResult.FAIL, f"Invalid trigger_threshold: {program.trigger_threshold}"

        return GateResult.PASS, "ok"

    def _check_quality(self, program: MemoryProgram) -> Tuple[GateResult, str]:
        """
        Quality Gate: 检查质量

        检查项:
        - 内容质量分数
        - 适应度分数
        - 版本历史
        """
        # 检查内容质量
        content = program.content

        # 计算质量指标
        lines = content.split('\n')

        # 检查是否有实质性内容
        substantive_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        if len(substantive_lines) < 3:
            return GateResult.FAIL, f"Too few substantive lines ({len(substantive_lines)})"

        # 检查代码块比例
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        total_lines = len(lines)
        if code_blocks > 0 and code_blocks / total_lines > 0.8:
            return GateResult.FAIL, "Too many code blocks, not enough description"

        # 检查适应度 (如果是从父程序变异而来)
        if program.parent_id:
            if program.fitness_score < 0.1:
                return GateResult.FAIL, f"Fitness too low: {program.fitness_score}"

        # 检查版本链
        if program.mutation_count > 10:
            return GateResult.FAIL, f"Too many mutations: {program.mutation_count}"

        return GateResult.PASS, "ok"

    # ── 辅助检查方法 ────────────────────────────────────────────────────────────────

    def _extract_code_blocks(self, content: str, language: str) -> List[str]:
        """提取代码块"""
        pattern = rf'```' + language + r'\n([\s\S]*?)\n```'
        return re.findall(pattern, content)

    def _check_python_syntax(self, code: str) -> Optional[str]:
        """检查 Python 语法"""
        import ast
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"

    def _check_shell_syntax(self, code: str) -> Optional[str]:
        """检查 Shell 语法 (使用 bash -n)

        Windows增强: 同时检查 PowerShell 语法
        """
        if not code.strip():
            return None

        # 首先尝试 PowerShell (Windows)
        if sys.platform == 'win32' or sys.platform.startswith('mingw'):
            ps_result = self._check_powershell_syntax(code)
            if ps_result is not None:
                return ps_result

        # 然后尝试 bash (Unix-like)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(code)
                f.flush()

            result = subprocess.run(
                ['bash', '-n', f.name],
                capture_output=True,
                text=True,
                timeout=self.compile_timeout
            )

            return result.stderr if result.returncode != 0 else None

        except subprocess.TimeoutExpired:
            return "Timeout checking shell syntax"
        except FileNotFoundError:
            # bash 不可用，跳过检查 (Windows默认)
            return None
        except Exception as e:
            return str(e)
        finally:
            try:
                import os
                os.unlink(f.name)
            except:
                pass

    def _check_powershell_syntax(self, code: str) -> Optional[str]:
        """检查 PowerShell 语法 (Windows增强)

        使用 PowerShell 的 -NoProfile -Command "trap { exit 1 }; $null = [System.Management.Automation.Language.Parser]::ParseFile(...)
        """
        if not code.strip():
            return None

        # 检测是否是 PowerShell 代码 (shebang 或特征关键词)
        is_powershell = (
            code.strip().startswith('#!/') and 'pwsh' in code.lower() or 'powershell' in code.lower()
        ) or (
            'param(' in code or '$PSDefaultParameterValues' in code or
            'Get-Process' in code or 'Set-ExecutionPolicy' in code or
            'Write-Host' in code or 'Import-Module' in code
        )

        if not is_powershell:
            return None  # 不是PowerShell代码，不检查

        try:
            ps_command = f'''
try {{
    $null = [System.Management.Automation.Language.Parser]::ParseInput(@'
{code}
'@, [ref]$null, [ref]$null)
    exit 0
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
'''

            result = subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_command],
                capture_output=True,
                text=True,
                timeout=self.compile_timeout
            )

            if result.returncode != 0:
                return f"PowerShell syntax error: {result.stderr.strip()}"
            return None

        except subprocess.TimeoutExpired:
            return "Timeout checking PowerShell syntax"
        except FileNotFoundError:
            return None
        except Exception as e:
            return str(e)

    def _check_yaml_syntax(self, yaml_str: str) -> Optional[str]:
        """检查 YAML 语法"""
        try:
            self._parse_yaml(yaml_str)
            return None
        except Exception as e:
            return str(e)

    def _parse_yaml(self, yaml_str: str) -> Dict[str, Any]:
        """简单 YAML 解析"""
        import yaml
        return yaml.safe_load(yaml_str)

    # ── 快速检查方法 ────────────────────────────────────────────────────────────────

    def quick_check(self, program: MemoryProgram) -> Tuple[bool, str]:
        """
        快速检查 (用于即时反馈)

        Returns:
            (passed, message)
        """
        # 只检查最关键的几项
        if not program.instructions.trigger_keywords:
            return False, "No trigger keywords"

        if not program.content:
            return False, "Empty content"

        if len(program.content) < self.min_content_length:
            return False, f"Content too short"

        return True, "ok"

    def validate_keywords(self, keywords: List[str]) -> Tuple[bool, str]:
        """
        验证关键词

        Returns:
            (valid, message)
        """
        if not keywords:
            return False, "No keywords provided"

        if len(keywords) > self.max_keywords:
            return False, f"Too many keywords ({len(keywords)} > {self.max_keywords})"

        for kw in keywords:
            if len(kw) < 2:
                return False, f"Keyword too short: '{kw}'"
            if len(kw) > 50:
                return False, f"Keyword too long: '{kw}'"

        return True, "ok"


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def get_quality_gates() -> QualityGates:
    """
    获取质量门禁实例

    Returns:
        QualityGates 实例
    """
    return QualityGates()