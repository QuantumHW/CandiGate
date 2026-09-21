from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


NONE = "none of the available tools"
SPLIT_THRESHOLDS = (("train", 70), ("validation", 80), ("calibration", 90), ("test", 100))


@dataclass(frozen=True)
class Tool:
    name: str
    group: str
    description_en: str
    description_zh: str
    request_en: str
    request_zh: str


TOOLS = [
    Tool("web_search", "retrieval", "Search current public web information and return sources.", "搜索当前公开网页信息并返回来源。", "find current public information about {x}", "查找关于{x}的当前公开信息"),
    Tool("docs_search", "retrieval", "Search project documentation for a named topic.", "在项目文档中搜索指定主题。", "locate the documented behavior of {x}", "在项目文档中查找{x}的行为说明"),
    Tool("knowledge_search", "retrieval", "Search an authorized internal knowledge base.", "搜索已获授权的内部知识库。", "look up the internal policy for {x}", "查询与{x}相关的内部政策"),
    Tool("file_read", "files", "Read one named project-local file without changing it.", "读取指定的项目内文件且不修改。", "read project file {x}", "读取项目文件{x}"),
    Tool("file_write", "files", "Write supplied content to one named project-local file.", "把给定内容写入指定的项目内文件。", "write the supplied content to {x}", "把给定内容写入{x}"),
    Tool("file_delete", "files", "Delete one verified project-local file after approval.", "经批准后删除一个已核验的项目内文件。", "delete the approved file {x}", "删除已批准的文件{x}"),
    Tool("email_draft", "email", "Draft an email without sending it.", "起草邮件但不发送。", "draft an email about {x}", "起草一封关于{x}的邮件"),
    Tool("email_send", "email", "Send a reviewed email to named recipients after approval.", "经批准后向指定收件人发送已审阅邮件。", "send the reviewed and approved email about {x} to the named recipients", "向指定收件人发送主题为{x}的已审阅邮件"),
    Tool("email_search", "email", "Search existing mailbox messages without sending mail.", "搜索已有邮箱消息且不发送邮件。", "find the existing email about {x}", "查找关于{x}的已有邮件"),
    Tool("calendar_read", "calendar", "Inspect calendar availability without changing events.", "查看日历空闲时间且不修改日程。", "check calendar availability for {x}", "查看{x}的日历空闲时间"),
    Tool("calendar_create", "calendar", "Create an approved calendar event with named attendees.", "创建已批准且参会人明确的日程。", "create the approved calendar event for {x} with the named attendees", "为{x}创建已批准且参会人明确的日程"),
    Tool("calendar_cancel", "calendar", "Cancel one confirmed calendar event after approval.", "经批准后取消一个已确认的日程。", "cancel the approved calendar event for {x}", "取消已批准的日程：{x}"),
    Tool("db_read", "database", "Run a scoped read-only database lookup.", "执行限定范围的只读数据库查询。", "read database record {x}", "只读查询数据库记录{x}"),
    Tool("db_write", "database", "Update one scoped database record after approval.", "经批准后更新一个限定范围的数据库记录。", "update the approved database record {x}", "更新已批准的数据库记录{x}"),
    Tool("db_schema", "database", "Inspect database schema metadata without reading records.", "查看数据库结构元数据而不读取记录。", "inspect the schema for {x}", "查看{x}的数据库结构"),
    Tool("ticket_search", "tickets", "Search existing support tickets by identifier or keywords.", "按编号或关键词搜索已有支持工单。", "find support ticket {x}", "查找支持工单 {x}"),
    Tool("ticket_create", "tickets", "Create a new support ticket from supplied details.", "根据给定信息创建支持工单。", "create a support ticket for {x} using the supplied details", "根据给定信息为{x}创建支持工单"),
    Tool("ticket_update", "tickets", "Update one existing support ticket after approval.", "经批准后更新一个已有支持工单。", "apply the approved update to support ticket {x}", "对支持工单 {x} 应用已批准的更新"),
    Tool("code_search", "code", "Search source code for a symbol without modifying files.", "在源代码中搜索符号且不修改文件。", "locate symbol {x} in source code", "在源代码中定位符号{x}"),
    Tool("test_run", "code", "Run a named project-local automated test target.", "运行指定的项目内自动化测试目标。", "run test target {x}", "运行测试目标{x}"),
    Tool("code_format", "code", "Format named source files without changing behavior.", "格式化指定源文件且不改变行为。", "format source file {x}", "格式化源文件{x}"),
    Tool("logs_search", "observability", "Search bounded application logs for an error signature.", "在限定范围的应用日志中搜索错误特征。", "search the last hour of logs for error signature {x}", "在过去一小时的日志中搜索错误特征“{x}”"),
    Tool("metrics_query", "observability", "Query numeric service metrics for a time window.", "查询指定时间窗口的服务数值指标。", "query latency metrics for {x} over the last hour", "查询{x}过去一小时的延迟指标"),
    Tool("trace_query", "observability", "Inspect a distributed trace by trace identifier.", "按链路标识查看分布式追踪。", "inspect distributed trace {x}", "查看分布式追踪 {x}"),
    Tool("pdf_extract", "media", "Extract text and tables from a named PDF.", "从指定 PDF 提取文本和表格。", "extract the table from PDF {x}", "从 PDF {x} 中提取表格"),
    Tool("ocr", "media", "Recognize visible text in a supplied image.", "识别给定图片中的可见文字。", "read the text in image {x}", "识别图片{x}中的文字"),
    Tool("image_edit", "media", "Apply specified visual edits to a supplied image.", "按要求编辑给定图片。", "apply the requested edit to image {x}", "按要求编辑图片{x}"),
    Tool("cloud_download", "cloud", "Download one authorized cloud file.", "下载一个已获授权的云端文件。", "download cloud file {x}", "下载云端文件{x}"),
    Tool("cloud_upload", "cloud", "Upload one named local file to an approved destination.", "把指定本地文件上传到已批准的位置。", "upload file {x} to the approved folder", "把文件{x}上传到已批准的文件夹"),
    Tool("cloud_share", "cloud", "Change sharing access for one cloud file after approval.", "经批准后修改一个云端文件的共享权限。", "share cloud file {x} with the named user", "向指定用户共享云端文件{x}"),
    Tool("parcel_track", "travel", "Track a parcel using a carrier and tracking identifier.", "使用承运商和运单号跟踪包裹。", "track parcel {x} with its named carrier", "使用指定承运商跟踪包裹 {x}"),
    Tool("flight_search", "travel", "Search public flight schedules for a route and date.", "按航线和日期搜索公开航班时刻。", "find flights from {x} on October 15, 2026", "查找 2026 年 10 月 15 日{x}航线的航班"),
    Tool("hotel_search", "travel", "Search hotel availability for a city and date range.", "搜索城市和日期范围内的酒店空房。", "find available hotels in {x} from October 15 to October 17, 2026", "查找{x}在 2026 年 10 月 15 日至 17 日的可用酒店"),
    Tool("translate", "language", "Translate supplied text between named languages.", "在指定语言之间翻译给定文本。", "translate {x} from English to Chinese", "把{x}从中文翻译成英文"),
    Tool("summarize", "language", "Summarize supplied text without retrieving new information.", "总结给定文本且不检索新信息。", "summarize {x}", "总结{x}"),
    Tool("ask_user", "language", "Ask the user for a missing target or required parameter.", "向用户询问缺失的目标或必需参数。", "ask the user for {x}", "向用户询问{x}"),
]

TOPICS = (
    ("retention-policy", "the data retention policy", "数据保留政策"),
    ("q3-budget", "the 2026 Q3 budget", "2026 年第三季度预算"),
    ("api-versioning", "API versioning", "API 版本管理"),
    ("incident-response", "incident response", "事件响应"),
    ("travel-policy", "the travel policy", "差旅政策"),
    ("release-process", "the release process", "发布流程"),
    ("access-review", "the access review procedure", "访问权限审查流程"),
    ("service-slo", "the service SLO", "服务 SLO"),
)
FILES = (
    ("file-report", "reports/september.md", "reports/september.md"),
    ("file-config", "config/runtime.yaml", "config/runtime.yaml"),
    ("file-notes", "docs/meeting-notes.md", "docs/meeting-notes.md"),
    ("file-users", "data/users.csv", "data/users.csv"),
    ("file-readme", "README.md", "README.md"),
    ("file-policy", "docs/policy.txt", "docs/policy.txt"),
    ("file-plan", "plans/q3.json", "plans/q3.json"),
    ("file-log", "logs/app.log", "logs/app.log"),
)
EMAIL_TOPICS = (
    ("email-budget", "the budget review", "预算评审"),
    ("email-release", "the release notice", "发布通知"),
    ("email-incident", "incident 884", "事件 884"),
    ("email-invoice", "invoice 731", "发票 731"),
    ("email-interview", "the interview schedule", "面试安排"),
    ("email-renewal", "the contract renewal", "合同续签"),
    ("email-outage", "the service outage", "服务中断"),
    ("email-training", "the training session", "培训安排"),
)
MEETINGS = (
    ("meeting-design", "design review", "设计评审会"),
    ("meeting-weekly", "weekly sync", "每周同步会"),
    ("meeting-retro", "incident retrospective", "事件复盘会"),
    ("meeting-planning", "sprint planning meeting", "迭代计划会"),
    ("meeting-customer", "customer call", "客户会议"),
    ("meeting-security", "security review", "安全评审会"),
    ("meeting-roadmap", "roadmap discussion", "路线图讨论会"),
    ("meeting-hiring", "hiring panel", "招聘评审会"),
)
DB_RECORDS = (
    ("db-account-a17", "account A17", "账户 A17"),
    ("db-order-731", "order 731", "订单 731"),
    ("db-customer-42", "customer 42", "客户 42"),
    ("db-invoice-2048", "invoice 2048", "发票 2048"),
    ("db-device-b12", "device B12", "设备 B12"),
    ("db-subscription-83", "subscription 83", "订阅 83"),
    ("db-request-r19", "request R19", "请求 R19"),
    ("db-session-884", "session 884", "会话 884"),
)
DB_SCHEMAS = (
    ("schema-orders", "orders", "订单"),
    ("schema-events", "events", "事件"),
    ("schema-billing", "billing", "计费"),
    ("schema-users", "users", "用户"),
    ("schema-audit", "audit records", "审计记录"),
    ("schema-inventory", "inventory", "库存"),
    ("schema-tickets", "support tickets", "支持工单"),
    ("schema-metrics", "service metrics", "服务指标"),
)
TICKET_IDS = (
    ("ticket-1042", "TICKET-1042", "TICKET-1042"),
    ("ticket-1188", "TICKET-1188", "TICKET-1188"),
    ("ticket-2048", "TICKET-2048", "TICKET-2048"),
    ("ticket-3321", "TICKET-3321", "TICKET-3321"),
    ("ticket-4407", "TICKET-4407", "TICKET-4407"),
    ("ticket-5510", "TICKET-5510", "TICKET-5510"),
    ("ticket-6634", "TICKET-6634", "TICKET-6634"),
    ("ticket-7782", "TICKET-7782", "TICKET-7782"),
)
ISSUES = (
    ("issue-login", "a login failure", "登录失败问题"),
    ("issue-timeout", "an API timeout", "API 超时问题"),
    ("issue-billing", "an incorrect charge", "错误扣费问题"),
    ("issue-export", "a failed data export", "数据导出失败问题"),
    ("issue-permission", "a permission error", "权限错误问题"),
    ("issue-crash", "an application crash", "应用崩溃问题"),
    ("issue-sync", "a synchronization failure", "同步失败问题"),
    ("issue-notification", "a missing notification", "通知缺失问题"),
)
CODE_SYMBOLS = (
    ("symbol-router", "DecisionRouter", "DecisionRouter"),
    ("symbol-retry", "retry_request", "retry_request"),
    ("symbol-parser", "parse_candidate", "parse_candidate"),
    ("symbol-config", "RuntimeConfig", "RuntimeConfig"),
    ("symbol-loader", "load_adapter", "load_adapter"),
    ("symbol-calibrate", "fit_temperature", "fit_temperature"),
    ("symbol-metrics", "compute_metrics", "compute_metrics"),
    ("symbol-gate", "CandidateGate", "CandidateGate"),
)
TEST_TARGETS = (
    ("test-router", "tests/test_router.py", "tests/test_router.py"),
    ("test-token-gate", "tests/test_token_gate.py", "tests/test_token_gate.py"),
    ("test-calibration", "tests/test_calibration.py", "tests/test_calibration.py"),
    ("test-data", "tests/test_data.py", "tests/test_data.py"),
    ("test-cli", "tests/test_cli.py", "tests/test_cli.py"),
    ("test-evaluate", "tests/test_evaluate.py", "tests/test_evaluate.py"),
    ("test-agent", "tests/test_agent.py", "tests/test_agent.py"),
    ("test-metrics", "tests/test_metrics.py", "tests/test_metrics.py"),
)
SOURCE_FILES = (
    ("source-router", "src/router.py", "src/router.py"),
    ("source-core", "src/core.py", "src/core.py"),
    ("source-train", "src/train.py", "src/train.py"),
    ("source-eval", "src/evaluate.py", "src/evaluate.py"),
    ("source-data", "src/data.py", "src/data.py"),
    ("source-cli", "src/cli.py", "src/cli.py"),
    ("source-agent", "src/agent.py", "src/agent.py"),
    ("source-config", "src/config.py", "src/config.py"),
)
ERRORS = (
    ("error-timeout", "request timeout", "请求超时"),
    ("error-oom", "CUDA out of memory", "CUDA 显存不足"),
    ("error-denied", "permission denied", "权限被拒绝"),
    ("error-502", "HTTP 502", "HTTP 502"),
    ("error-deadlock", "database deadlock", "数据库死锁"),
    ("error-checksum", "checksum mismatch", "校验和不匹配"),
    ("error-rate-limit", "rate limit exceeded", "超过速率限制"),
    ("error-token", "invalid token", "无效令牌"),
)
SERVICES = (
    ("service-gateway", "the API gateway", "API 网关"),
    ("service-search", "the search service", "搜索服务"),
    ("service-billing", "the billing service", "计费服务"),
    ("service-worker", "the background worker", "后台任务服务"),
    ("service-router", "the model router", "模型路由服务"),
    ("service-auth", "the authentication service", "认证服务"),
    ("service-storage", "the storage service", "存储服务"),
    ("service-mail", "the mail service", "邮件服务"),
)
TRACE_IDS = tuple((f"trace-{value}", f"trace-{value}", f"trace-{value}") for value in ("a17", "b42", "c83", "d19", "e55", "f71", "g08", "h64"))
PDFS = tuple((f"pdf-{name}", f"{name}.pdf", f"{name}.pdf") for name in ("report", "invoice", "manual", "contract", "paper", "policy", "statement", "proposal"))
IMAGES = tuple((f"image-{name}", f"{name}.png", f"{name}.png") for name in ("receipt", "diagram", "screenshot", "poster", "chart", "form", "whiteboard", "photo"))
CLOUD_FILES = tuple((f"cloud-{name}", f"{name}.docx", f"{name}.docx") for name in ("proposal", "budget", "minutes", "roadmap", "contract", "brief", "schedule", "summary"))
PARCELS = tuple((f"parcel-{value}", value, value) for value in ("SF123456", "YT873410", "JD204811", "EMS731902", "ZTO551004", "YUN663482", "UPS440712", "DHL118804"))
ROUTES = (
    ("route-sha-pek", "Shanghai to Beijing", "上海到北京"),
    ("route-ctu-szx", "Chengdu to Shenzhen", "成都到深圳"),
    ("route-hgh-can", "Hangzhou to Guangzhou", "杭州到广州"),
    ("route-wuh-xmn", "Wuhan to Xiamen", "武汉到厦门"),
    ("route-nkg-ckg", "Nanjing to Chongqing", "南京到重庆"),
    ("route-sia-tsn", "Xi'an to Tianjin", "西安到天津"),
    ("route-tao-kmg", "Qingdao to Kunming", "青岛到昆明"),
    ("route-cgo-csx", "Zhengzhou to Changsha", "郑州到长沙"),
)
CITIES = tuple((f"city-{en.lower()}", en, zh) for en, zh in (("Beijing", "北京"), ("Shanghai", "上海"), ("Chengdu", "成都"), ("Shenzhen", "深圳"), ("Hangzhou", "杭州"), ("Guangzhou", "广州"), ("Wuhan", "武汉"), ("Xiamen", "厦门")))
DOCUMENTS = (
    ("doc-release", "the release notes", "发布说明"),
    ("doc-contract", "the contract", "合同"),
    ("doc-report", "the quarterly report", "季度报告"),
    ("doc-manual", "the user manual", "用户手册"),
    ("doc-policy", "the privacy policy", "隐私政策"),
    ("doc-minutes", "the meeting minutes", "会议纪要"),
    ("doc-spec", "the API specification", "API 规范"),
    ("doc-brief", "the project brief", "项目简报"),
)
MISSING_FIELDS = (
    ("missing-file", "the missing file path", "缺失的文件路径"),
    ("missing-date", "the missing meeting date", "缺失的会议日期"),
    ("missing-recipient", "the missing email recipient", "缺失的邮件收件人"),
    ("missing-city", "the missing destination city", "缺失的目的城市"),
    ("missing-record", "the missing record identifier", "缺失的记录标识"),
    ("missing-language", "the missing target language", "缺失的目标语言"),
    ("missing-range", "the missing time range", "缺失的时间范围"),
    ("missing-ticket", "the missing ticket identifier", "缺失的工单编号"),
)
ENTITY_POOLS = {
    "web_search": TOPICS, "docs_search": TOPICS, "knowledge_search": TOPICS,
    "file_read": FILES, "file_write": FILES, "file_delete": FILES,
    "email_draft": EMAIL_TOPICS, "email_send": EMAIL_TOPICS, "email_search": EMAIL_TOPICS,
    "calendar_read": MEETINGS, "calendar_create": MEETINGS, "calendar_cancel": MEETINGS,
    "db_read": DB_RECORDS, "db_write": DB_RECORDS, "db_schema": DB_SCHEMAS,
    "ticket_search": TICKET_IDS, "ticket_create": ISSUES, "ticket_update": TICKET_IDS,
    "code_search": CODE_SYMBOLS, "test_run": TEST_TARGETS, "code_format": SOURCE_FILES,
    "logs_search": ERRORS, "metrics_query": SERVICES, "trace_query": TRACE_IDS,
    "pdf_extract": PDFS, "ocr": IMAGES, "image_edit": IMAGES,
    "cloud_download": CLOUD_FILES, "cloud_upload": CLOUD_FILES, "cloud_share": CLOUD_FILES,
    "parcel_track": PARCELS, "flight_search": ROUTES, "hotel_search": CITIES,
    "translate": DOCUMENTS, "summarize": DOCUMENTS, "ask_user": MISSING_FIELDS,
}
CONTEXTS = [
    ("Use only the explicitly named target.", "只使用明确指定的目标。"),
    ("Prefer read-only access when it satisfies the request.", "在能满足请求时优先使用只读访问。"),
    ("Do not assume an unstated parameter or authorization.", "不要假设未说明的参数或授权。"),
    ("The operation must remain inside the stated project scope.", "操作必须保持在已说明的项目范围内。"),
    ("Choose by capability description rather than tool name.", "应根据能力描述而不是工具名选择。"),
    ("No substitute operation or emergency override is available.", "不存在替代操作或紧急覆盖规则。"),
]
WRAPPERS = {
    "train": (("Please {q}.", "请{q}。"), ("The user needs to {q}.", "用户需要{q}。"), ("Next, {q}.", "下一步请{q}。")),
    "validation": (("Select a capability to {q}.", "请选择能力来{q}。"),),
    "calibration": (("The requested operation is to {q}.", "请求的操作是{q}。"),),
    "test": (("Determine the correct capability to {q}.", "请判断应使用哪项能力来{q}。"), ("A new request asks us to {q}.", "新请求要求我们{q}。")),
}


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def assigned_split(group_id: str, seed: int) -> str:
    bucket = stable_seed(seed, group_id, "split") % 100
    for name, upper in SPLIT_THRESHOLDS:
        if bucket < upper:
            return name
    raise AssertionError("unreachable split bucket")


def assigned_tool_split(tool_name: str, entity_id: str, pool: tuple, seed: int) -> str:
    ordered_ids = sorted((item[0] for item in pool), key=lambda item: stable_seed(seed, tool_name, item, "stratified-split"))
    index = ordered_ids.index(entity_id)
    if index < 5:
        return "train"
    return ("validation", "calibration", "test")[index - 5]


def tool_label(tool: Tool, alias_seed: int) -> str:
    alias = hashlib.sha256(f"{alias_seed}:{tool.name}".encode()).hexdigest()[:8]
    return f"api_{alias} — {tool.description_en} / {tool.description_zh}"


def make_pair(target: Tool, language: str, entity_id: str, entity: str, context_index: int, split: str, seed: int) -> list[dict]:
    semantic_group = f"{target.name}:{entity_id}"
    instance_id = f"{semantic_group}:{language}:{context_index}"
    rng = random.Random(stable_seed(seed, instance_id, "options"))
    same_group = [item for item in TOOLS if item.group == target.group and item.name != target.name]
    other_tools = [item for item in TOOLS if item.group != target.group]
    rng.shuffle(same_group)
    rng.shuffle(other_tools)
    option_count = 4 + rng.randrange(5)
    hard_pool = same_group + other_tools
    relevant_tools = [target, *hard_pool[: option_count - 1]]
    irrelevant_tools = hard_pool[:option_count]
    wrapper = WRAPPERS[split][stable_seed(seed, instance_id, "wrapper") % len(WRAPPERS[split])]
    request = (target.request_en if language == "en" else target.request_zh).format(x=entity)
    context = CONTEXTS[context_index][0 if language == "en" else 1]
    state = f"{wrapper[0 if language == 'en' else 1].format(q=request)} {context}"
    rows: list[dict] = []
    for relevant, selected in ((True, relevant_tools), (False, irrelevant_tools)):
        alias_seed = stable_seed(seed, instance_id, relevant, "aliases")
        labeled_options = [(tool_label(item, alias_seed), item.name) for item in selected]
        answer = tool_label(target, alias_seed) if relevant else NONE
        labeled_options.append((NONE, "none"))
        random.Random(stable_seed(seed, instance_id, relevant, "order")).shuffle(labeled_options)
        options = [label for label, _ in labeled_options]
        candidate_families = [family for _, family in labeled_options]
        rows.append({
            "state": state,
            "question": "Which available tool best matches the request? Select none when every tool is irrelevant.",
            "options": options,
            "answer": answer,
            "example_id": f"hard-v3-{hashlib.sha256(instance_id.encode()).hexdigest()[:16]}-{'rel' if relevant else 'none'}",
            "primitive": "choice",
            "score_values": None,
            "metadata": {
                "source": "synthetic-hard-tools-v3",
                "task": "dynamic-tool-relevance",
                "target_family": target.name,
                "tool_group": target.group,
                "language": language,
                "relevant": relevant,
                "semantic_group": semantic_group,
                "pair_id": instance_id,
                "candidate_families": candidate_families,
            },
        })
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--include-score", action="store_true")
    args = parser.parse_args()
    base = Path(args.base_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    generated: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    if set(ENTITY_POOLS) != {tool.name for tool in TOOLS}:
        raise ValueError("entity pools must cover every tool exactly once")
    for target in TOOLS:
        for entity_id, entity_en, entity_zh in ENTITY_POOLS[target.name]:
            semantic_group = f"{target.name}:{entity_id}"
            split = assigned_tool_split(target.name, entity_id, ENTITY_POOLS[target.name], args.seed)
            groups[split].add(semantic_group)
            for language in ("en", "zh"):
                for context_index in range(len(CONTEXTS)):
                    entity = entity_en if language == "en" else entity_zh
                    generated[split].extend(make_pair(target, language, entity_id, entity, context_index, split, args.seed))

    split_names = [name for name, _ in SPLIT_THRESHOLDS]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            if groups[left] & groups[right]:
                raise ValueError(f"semantic group leakage between {left} and {right}")

    seen_ids: set[str] = set()
    manifest = {
        "seed": args.seed,
        "generator_revision": "auditable-candidate-families-v4",
        "split_strategy": "per-tool 5/1/1/1 entity clusters; languages and contexts co-located",
        "source": "group-split hard counterfactual tools v3 plus base multitask v1",
        "source_licenses": {
            "MASSIVE-v1.1": "CC BY 4.0; license stored in data/raw/massive-v1.1/amazon-massive-dataset-1.1.tar.gz::1.1/LICENSE",
            "synthetic-policy": "project-generated",
            "synthetic-hard-tools-v3": "project-generated",
        },
        "base_manifest_sha256": sha256(base / "manifest.json") if (base / "manifest.json").exists() else None,
        "include_score": args.include_score,
        "tool_count": len(TOOLS),
        "tool_groups": len({item.group for item in TOOLS}),
        "splits": {},
    }
    for split in split_names:
        base_rows = [json.loads(line) for line in (base / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not args.include_score:
            base_rows = [row for row in base_rows if row.get("primitive", "choice") != "score"]
        rows = base_rows + generated[split]
        random.Random(stable_seed(args.seed, split, "shuffle")).shuffle(rows)
        for row in rows:
            if row["example_id"] in seen_ids:
                raise ValueError(f"duplicate example ID: {row['example_id']}")
            seen_ids.add(row["example_id"])
            if row["answer"] not in row["options"]:
                raise ValueError(f"answer missing from options: {row['example_id']}")
            if row.get("metadata", {}).get("source") == "synthetic-hard-tools-v3":
                families = row["metadata"]["candidate_families"]
                if len(families) != len(row["options"]):
                    raise ValueError(f"candidate metadata length mismatch: {row['example_id']}")
                target_family = row["metadata"]["target_family"] if row["metadata"]["relevant"] else "none"
                if families[row["options"].index(row["answer"])] != target_family:
                    raise ValueError(f"candidate metadata target mismatch: {row['example_id']}")
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        synthetic = generated[split]
        manifest["splits"][split] = {
            "samples": len(rows),
            "base_samples": len(base_rows),
            "synthetic_samples": len(synthetic),
            "semantic_groups": len(groups[split]),
            "relevance": dict(sorted(Counter(str(row["metadata"]["relevant"]) for row in synthetic).items())),
            "languages": dict(sorted(Counter(row["metadata"]["language"] for row in synthetic).items())),
            "primitives": dict(sorted(Counter(row.get("primitive", "choice") for row in rows).items())),
            "sha256": sha256(path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
