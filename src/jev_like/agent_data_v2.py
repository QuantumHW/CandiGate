from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


SPLIT_COUNTS = {"train": 4800, "validation": 600, "calibration": 600, "test": 1200}
NONE = "none of the available tools"
TOOLS = [
    ("weather", "Retrieve a public weather forecast for a specified place and date.", "查询指定地点和日期的公开天气预报。", "check the forecast for {x}", "查询{x}的天气预报"),
    ("currency", "Convert an amount between two currencies using a quoted exchange rate.", "按给定汇率换算两种货币金额。", "convert the stated amount of {x} into another currency", "把{x}的指定金额换算成另一种货币"),
    ("parcel", "Track a parcel using its carrier and tracking identifier.", "使用承运商和运单号跟踪包裹。", "track parcel {x}", "跟踪包裹{x}"),
    ("flight", "Search public flight schedules by airports and travel date.", "按机场和日期搜索公开航班时刻。", "find flights for route {x}", "查找{x}航线的航班"),
    ("hotel", "Search hotel availability for a city and date range.", "搜索城市和日期范围内的酒店空房。", "find available hotels in {x}", "查找{x}的可用酒店"),
    ("pdf", "Extract text and tables from a named local PDF document.", "从指定本地 PDF 中提取文本和表格。", "extract the table from PDF {x}", "从 PDF {x} 中提取表格"),
    ("ocr", "Recognize visible text in a supplied image.", "识别给定图片中的可见文字。", "read the text in image {x}", "识别图片{x}里的文字"),
    ("translate", "Translate supplied text between named languages.", "在指定语言之间翻译给定文本。", "translate document {x} into the requested language", "把文档{x}翻译成指定语言"),
    ("code_search", "Search source files for a symbol or exact text without modifying them.", "在源文件中搜索符号或文本且不修改文件。", "locate symbol {x} in the repository", "在仓库中定位符号{x}"),
    ("test", "Run a specified project-local automated test target.", "运行指定的项目内自动化测试目标。", "run test target {x}", "运行测试目标{x}"),
    ("logs", "Search application logs in a bounded time range for an error signature.", "在限定时间范围内按错误特征搜索应用日志。", "search recent logs for error {x}", "在近期日志中搜索错误{x}"),
    ("metrics", "Query numeric service metrics for a named service and time window.", "查询指定服务和时间窗口的数值指标。", "query latency metrics for service {x}", "查询服务{x}的延迟指标"),
    ("db_read", "Run a scoped read-only lookup against a structured database.", "对结构化数据库执行限定范围的只读查询。", "look up read-only record {x}", "只读查询记录{x}"),
    ("calendar_read", "Inspect calendar availability without creating or changing events.", "只读查看日历空闲时间，不创建或修改日程。", "check calendar availability for {x}", "查看{x}的日历空闲时间"),
    ("calendar_create", "Create a calendar event with specified time and attendees.", "按指定时间和参会人创建日程。", "create the approved meeting {x}", "创建已批准的会议{x}"),
    ("email_draft", "Draft an email without sending it to external recipients.", "起草邮件但不向外部收件人发送。", "draft an email about {x}", "起草一封关于{x}的邮件"),
    ("email_send", "Send an approved email to explicitly named recipients.", "向明确指定的收件人发送已批准的邮件。", "send the approved email {x}", "发送已批准的邮件{x}"),
    ("ticket_search", "Search existing support tickets by identifier or keywords.", "按编号或关键词搜索已有支持工单。", "find support ticket {x}", "查找支持工单{x}"),
    ("ticket_create", "Create a new support ticket with a title and description.", "使用标题和描述创建新的支持工单。", "create a support ticket for {x}", "为{x}创建支持工单"),
    ("file_read", "Read one named project-local text file without changing it.", "读取一个指定的项目内文本文件且不修改。", "read local file {x}", "读取本地文件{x}"),
    ("file_write", "Write supplied content to a named project-local file.", "把给定内容写入指定的项目内文件。", "write the supplied content to file {x}", "把给定内容写入文件{x}"),
    ("calculate", "Evaluate an exact arithmetic expression locally.", "在本地精确计算算术表达式。", "calculate expression {x}", "计算表达式{x}"),
    ("web", "Search current public web information and return cited sources.", "搜索最新公开网页信息并返回来源。", "search current public information about {x}", "搜索关于{x}的最新公开信息"),
    ("ask", "Ask the user for a missing target, constraint, or required parameter.", "向用户询问缺失的目标、约束或必需参数。", "clarify the missing target for {x}", "询问{x}中缺失的目标"),
]
ENTITIES = ["A17", "project Orion", "the September report", "case 2048", "service delta", "订单 731", "项目晨星", "记录 B-42", "the named customer", "tomorrow morning", "2026-Q3", "module atlas"]
CONTEXTS = [
    ("Use only the explicitly named target.", "只使用明确指定的目标。"),
    ("Prefer a read-only operation when that satisfies the request.", "在能满足请求时优先使用只读操作。"),
    ("Do not assume any unstated parameter.", "不要假设任何未说明的参数。"),
    ("The result must be auditable from the request.", "结果必须能从请求中审计。"),
    ("No substitute operation has been authorized.", "没有授权替代操作。"),
    ("Choose according to capability descriptions, not tool names.", "应根据能力描述而非工具名选择。"),
    ("The standard project scope applies.", "采用标准项目范围。"),
    ("No emergency override is in effect.", "当前没有紧急覆盖规则。"),
]
WRAPPERS = {
    "train": [("Please {q}.", "请{q}。"), ("The user needs to {q}.", "用户需要{q}。"), ("Next, {q}.", "下一步请{q}。")],
    "validation": [("Select a tool to {q}.", "请选择工具来{q}。")],
    "calibration": [("The requested operation is to {q}.", "请求的操作是{q}。")],
    "test": [("Determine how to {q}.", "请判断如何{q}。"), ("A new request asks us to {q}.", "新请求要求我们{q}。")],
}


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def tool_label(tool: tuple[str, str, str, str, str], alias_seed: int) -> str:
    alias = hashlib.sha256(f"{alias_seed}:{tool[0]}".encode()).hexdigest()[:8]
    return f"api_{alias} — {tool[1]} / {tool[2]}"


def dynamic_row(split: str, index: int, seed: int, combo: tuple[int, int, int, int, int, int]) -> dict:
    rng = random.Random(stable_seed(seed, split, index))
    tool_index, language_index, entity_index, context_index, wrapper_index, relevant_index = combo
    target = TOOLS[tool_index]
    language = "en" if language_index == 0 else "zh"
    entity = ENTITIES[entity_index]
    phrase = target[3] if language == "en" else target[4]
    query = phrase.format(x=entity)
    wrappers = WRAPPERS[split]
    wrapper = wrappers[wrapper_index][0 if language == "en" else 1]
    context = CONTEXTS[context_index][0 if language == "en" else 1]
    state = f"{wrapper.format(q=query)} {context}"
    relevant = relevant_index == 1
    option_count = 2 + rng.randrange(7)
    distractors = [tool for tool in TOOLS if tool[0] != target[0]]
    rng.shuffle(distractors)
    selected = distractors[: option_count - (1 if relevant else 0)]
    if relevant:
        selected.append(target)
    alias_seed = stable_seed(seed, split, index, "aliases")
    labels = [tool_label(tool, alias_seed) for tool in selected]
    answer = tool_label(target, alias_seed) if relevant else NONE
    labels.append(NONE)
    rng.shuffle(labels)
    return {
        "state": state,
        "question": "Which available tool best matches the request? Select none when every tool is irrelevant.",
        "options": labels,
        "answer": answer,
        "example_id": f"dynamic-v2-{split}-{index:06d}",
        "primitive": "choice",
        "score_values": None,
        "metadata": {"source": "synthetic-dynamic-tools-v2", "task": "dynamic-tool-relevance",
                     "target_family": target[0], "language": language, "relevant": relevant,
                     "template_group": f"{split}:{target[0]}:{language}"},
    }


def make_dynamic_rows(split: str, count: int, seed: int) -> list[dict]:
    combinations = list(itertools.product(
        range(len(TOOLS)), range(2), range(len(ENTITIES)), range(len(CONTEXTS)),
        range(len(WRAPPERS[split])), range(2),
    ))
    random.Random(stable_seed(seed, split, "semantic-combinations")).shuffle(combinations)
    if count > len(combinations):
        raise ValueError(f"Requested {count} rows but only {len(combinations)} unique combinations exist")
    return [dynamic_row(split, index, seed, combo) for index, combo in enumerate(combinations[:count])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    base = Path(args.base_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": args.seed, "source": "base multitask v1 plus synthetic dynamic tools v2", "splits": {}}
    seen_ids: set[str] = set()
    for split, extra_count in SPLIT_COUNTS.items():
        base_rows = [json.loads(line) for line in (base / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        extra_rows = make_dynamic_rows(split, extra_count, args.seed)
        rows = base_rows + extra_rows
        random.Random(stable_seed(args.seed, split, "shuffle")).shuffle(rows)
        for row in rows:
            if row["example_id"] in seen_ids:
                raise ValueError(f"Duplicate example ID: {row['example_id']}")
            seen_ids.add(row["example_id"])
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        sources = Counter(row.get("metadata", {}).get("source", "unknown") for row in rows)
        manifest["splits"][split] = {"samples": len(rows), "base_samples": len(base_rows),
                                      "dynamic_samples": len(extra_rows), "sources": dict(sorted(sources.items())),
                                      "sha256": sha256(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
