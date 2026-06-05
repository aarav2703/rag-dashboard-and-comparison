from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


GRAPH_BUILD_VERSION = "graph-cache-v1"
GRAPH_ARTIFACT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
SKIP_RELATIONSHIP_TYPES = {"mentions", "supports", "located_in", "evidence_for"}
STOP_ENTITIES = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Figure",
    "Table",
    "Chapter",
    "Section",
    "Example",
    "Remark",
    "Definition",
    "Proof",
    "Page",
    "PDF",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def corpus_signature(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(chunks)).encode("utf-8"))
    for chunk in chunks:
        digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip(".,:;()[]{}")


def normalize_entity_label(value: Any) -> str:
    value = re.sub(r"^\[|\]$", "", str(value or "")).strip().lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return " ".join(value.split())


def graph_node_id(kind: str, value: Any) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).lower()).strip("_")
    return f"{kind}:{safe}"


def extract_chunk_title(text: str) -> str:
    match = re.match(r"\s*\[([^\]]{2,120})\]", str(text or ""))
    return match.group(1).strip() if match else ""


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(value)
    return ordered


def extract_entities(text: str, max_entities: int = 12) -> list[str]:
    candidates = []
    patterns = [
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b",
        r"\b(?:neural network|decision tree|support vector machine|gradient descent|linear regression|logistic regression|deep learning|machine learning|random forest|nearest neighbor|principal component analysis|bayesian network|markov chain|attention mechanism|transformer model)s?\b",
    ]
    for pattern in patterns:
        flags = re.IGNORECASE if pattern.startswith("\\b(?:") else 0
        for match in re.finditer(pattern, text, flags=flags):
            entity = canonical_entity(match.group(0))
            if len(entity) < 3 or entity in STOP_ENTITIES:
                continue
            if entity.lower() in {"and", "for", "with", "from", "where", "which", "there"}:
                continue
            candidates.append(entity.title() if entity.islower() else entity)

    unique = []
    seen = set()
    for entity in candidates:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            unique.append(entity)
        if len(unique) >= max_entities:
            break
    return unique


def extract_claims(text: str, max_claims: int = 3) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    claims = []
    for sentence in sentences:
        clean = sentence.strip()
        if len(clean) < 60 or len(clean) > 260:
            continue
        lower = clean.lower()
        if any(marker in lower for marker in (" is ", " are ", " means ", " represents ", " depends ", " uses ", " measures ", " computes ", " predicts ")):
            claims.append(clean)
        if len(claims) >= max_claims:
            break
    return claims


def fallback_relationships(chunks: list[dict[str, Any]], max_rows: int = 60) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        entities = extract_entities(chunk.get("chunk_text", ""), max_entities=8)
        for left_index, left in enumerate(entities[:5]):
            for right in entities[left_index + 1 : 5]:
                rows.append(
                    {
                        "source_entity": left,
                        "relationship": "co_occurs_with",
                        "target_entity": right,
                        "evidence_chunk_id": chunk["chunk_id"],
                        "confidence": 0.45,
                    }
                )
                if len(rows) >= max_rows:
                    return rows
    return rows


def build_relationship_communities(graph: dict[str, Any], max_communities: int = 8) -> list[dict[str, Any]]:
    entity_edges = [
        edge for edge in graph.get("edges", [])
        if str(edge.get("source", "")).startswith("entity:") and str(edge.get("target", "")).startswith("entity:")
    ]
    adjacency: dict[str, set[str]] = {}
    for edge in entity_edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
        adjacency.setdefault(edge["target"], set()).add(edge["source"])
    seen = set()
    communities = []
    nodes_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    for node_id in adjacency:
        if node_id in seen:
            continue
        stack = [node_id]
        component = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        labels = [nodes_by_id.get(item, {}).get("label", item) for item in component]
        communities.append(
            {
                "id": f"community:{len(communities) + 1}",
                "label": ", ".join(labels[:3]) or f"Community {len(communities) + 1}",
                "node_ids": component,
                "entity_labels": labels,
            }
        )
    return sorted(communities, key=lambda item: -len(item["node_ids"]))[:max_communities]


def summarize_graph_communities_fallback(communities: list[dict[str, Any]], max_items: int = 6) -> list[dict[str, Any]]:
    rows = []
    for community in communities[:max_items]:
        labels = community.get("entity_labels", [])
        rows.append(
            {
                "community_id": community["id"],
                "title": community["label"],
                "summary": f"Entities: {', '.join(labels[:6])}",
                "node_ids": community["node_ids"],
            }
        )
    return rows


def build_document_graph_cache(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    entity_to_sections: dict[str, set[str]] = {}
    section_entities: dict[str, list[str]] = {}
    title_to_sections: dict[str, set[str]] = {}

    document_id = "document:pdf"
    nodes[document_id] = {"id": document_id, "label": "Document", "type": "document", "weight": len(chunks)}
    relationships = fallback_relationships(chunks)

    for chunk in chunks:
        section_id = graph_node_id("section", chunk["chunk_id"])
        section_label = f"Page {chunk['page_number']} / chunk {chunk.get('chunk_index', 0) + 1}"
        nodes[section_id] = {
            "id": section_id,
            "label": section_label,
            "type": "section",
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "preview": chunk.get("preview", ""),
            "weight": 1,
            "is_focus": False,
        }
        edges[(document_id, section_id, "located_in")] = {
            "source": document_id,
            "target": section_id,
            "type": "located_in",
            "weight": 1,
        }

        chunk_title = extract_chunk_title(chunk["chunk_text"])
        entities = extract_entities(chunk["chunk_text"])
        if chunk_title:
            entities = unique_preserve_order([chunk_title, *entities])
            title_to_sections.setdefault(normalize_entity_label(chunk_title), set()).add(section_id)
        section_entities[section_id] = entities
        for entity in entities:
            entity_id = graph_node_id("entity", entity)
            nodes.setdefault(entity_id, {"id": entity_id, "label": entity, "type": "entity", "weight": 0})
            nodes[entity_id]["weight"] += 1
            entity_to_sections.setdefault(entity_id, set()).add(section_id)
            edges[(entity_id, section_id, "mentions")] = {
                "source": entity_id,
                "target": section_id,
                "type": "mentions",
                "weight": 1,
            }

        for claim_index, claim in enumerate(extract_claims(chunk["chunk_text"])):
            claim_id = graph_node_id("claim", f"{chunk['chunk_id']}:{claim_index}")
            nodes[claim_id] = {
                "id": claim_id,
                "label": claim[:96],
                "type": "claim",
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "preview": claim,
                "weight": 1,
                "is_focus": False,
            }
            edges[(claim_id, section_id, "supports")] = {
                "source": claim_id,
                "target": section_id,
                "type": "supports",
                "weight": 1,
            }
            for entity in entities[:6]:
                edges[(graph_node_id("entity", entity), claim_id, "mentions")] = {
                    "source": graph_node_id("entity", entity),
                    "target": claim_id,
                    "type": "mentions",
                    "weight": 1,
                }

        for left_index, left in enumerate(entities[:8]):
            for right in entities[left_index + 1 : 8]:
                left_id = graph_node_id("entity", left)
                right_id = graph_node_id("entity", right)
                key = tuple(sorted([left_id, right_id])) + ("co_occurs",)
                edges.setdefault(
                    key,
                    {"source": key[0], "target": key[1], "type": "co-occurs", "weight": 0},
                )
                edges[key]["weight"] += 1

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for rel in relationships:
        source = rel["source_entity"]
        target = rel["target_entity"]
        relation = rel["relationship"] or "related_to"
        evidence_chunk_id = rel["evidence_chunk_id"]
        source_id = graph_node_id("entity", source)
        target_id = graph_node_id("entity", target)
        for entity_id, label in ((source_id, source), (target_id, target)):
            nodes.setdefault(entity_id, {"id": entity_id, "label": label, "type": "entity", "weight": 0})
            nodes[entity_id]["weight"] += 1
        edges[(source_id, target_id, relation)] = {
            "source": source_id,
            "target": target_id,
            "type": relation,
            "weight": max(1, int(round(rel.get("confidence", 0.7) * 3))),
            "evidence_chunk_id": evidence_chunk_id,
            "confidence": rel.get("confidence", 0.7),
        }
        section_id = graph_node_id("section", evidence_chunk_id)
        if evidence_chunk_id in chunks_by_id:
            entity_to_sections.setdefault(source_id, set()).add(section_id)
            entity_to_sections.setdefault(target_id, set()).add(section_id)
            edges[(source_id, section_id, "evidence_for")] = {
                "source": source_id,
                "target": section_id,
                "type": "evidence_for",
                "weight": 1,
            }
            edges[(target_id, section_id, "evidence_for")] = {
                "source": target_id,
                "target": section_id,
                "type": "evidence_for",
                "weight": 1,
            }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "entity_to_sections": {key: sorted(value) for key, value in entity_to_sections.items()},
        "section_entities": section_entities,
        "title_to_sections": {key: sorted(value) for key, value in title_to_sections.items()},
        "relationships": relationships,
    }


def build_graph_artifacts(store_dir: Path, chunks: list[dict[str, Any]], source: str = "unknown") -> dict[str, Any]:
    started = time.perf_counter()
    graph = build_document_graph_cache(chunks)
    communities = build_relationship_communities(graph)
    community_summaries = summarize_graph_communities_fallback(communities)
    metadata = {
        "graph_build_version": GRAPH_BUILD_VERSION,
        "source": source,
        "chunk_count": len(chunks),
        "corpus_signature": corpus_signature(chunks),
        "external_tools_enabled": False,
        "relationship_source": "deterministic_fallback",
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "community_count": len(communities),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(store_dir / "graph.json", graph)
    write_json(
        store_dir / "graph_communities.json",
        {"communities": communities, "community_summaries": community_summaries},
    )
    write_json(store_dir / "graph_metadata.json", metadata)
    return metadata


def load_graph_artifacts(store_dir: Path, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    graph_path = store_dir / "graph.json"
    communities_path = store_dir / "graph_communities.json"
    metadata_path = store_dir / "graph_metadata.json"
    if not graph_path.exists() or not communities_path.exists() or not metadata_path.exists():
        raise RuntimeError("Graph cache is missing. Rebuild the corpus store.")

    metadata = load_json(metadata_path)
    expected_signature = corpus_signature(chunks)
    if metadata.get("corpus_signature") != expected_signature or metadata.get("chunk_count") != len(chunks):
        raise RuntimeError("Graph cache is stale. Rebuild the corpus store.")
    cache_key = (str(store_dir.resolve()), expected_signature)
    cached = GRAPH_ARTIFACT_CACHE.get(cache_key)
    if cached:
        return cached

    graph = load_json(graph_path)
    communities_payload = load_json(communities_path)
    payload = {
        **graph,
        "communities": communities_payload.get("communities", []),
        "community_summaries": communities_payload.get("community_summaries", []),
        "graph_metadata": metadata,
    }
    nodes_by_id = {node["id"]: node for node in payload.get("nodes", [])}
    relationship_adjacency: dict[str, list[dict[str, Any]]] = {}
    edge_adjacency: dict[str, list[dict[str, Any]]] = {}
    support_edges_by_target: dict[str, list[dict[str, Any]]] = {}
    for edge in payload.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        edge_adjacency.setdefault(source, []).append(edge)
        edge_adjacency.setdefault(target, []).append(edge)
        if edge.get("type") not in SKIP_RELATIONSHIP_TYPES:
            relationship_adjacency.setdefault(source, []).append(edge)
            relationship_adjacency.setdefault(target, []).append(edge)
        if edge.get("type") == "supports":
            support_edges_by_target.setdefault(target, []).append(edge)
    payload["_nodes_by_id"] = nodes_by_id
    payload["_entity_nodes"] = [node for node in payload.get("nodes", []) if node.get("type") == "entity"]
    payload["_relationship_adjacency"] = relationship_adjacency
    payload["_edge_adjacency"] = edge_adjacency
    payload["_support_edges_by_target"] = support_edges_by_target
    GRAPH_ARTIFACT_CACHE.clear()
    GRAPH_ARTIFACT_CACHE[cache_key] = payload
    return payload
