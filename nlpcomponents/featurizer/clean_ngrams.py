import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

class STSNgramCleaner:

    def __init__(
        self,
        auto_ngrams_path: Path,
        overlap_analysis_path: Path,
        dominance_ratio: float = 2.0,
        min_ngrams_per_tag: int = 5,
        use_tfidf: bool = False,
        strict_dominance_ratio: float = 4.0,
        confused_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        self.auto_ngrams_path = auto_ngrams_path
        self.overlap_analysis_path = overlap_analysis_path
        self.dominance_ratio = dominance_ratio
        self.min_ngrams_per_tag = min_ngrams_per_tag
        self.use_tfidf = use_tfidf
        self.strict_dominance_ratio = strict_dominance_ratio
        self.confused_pairs = set()
        if confused_pairs:
            for t1, t2 in confused_pairs:
                self.confused_pairs.add(frozenset([t1, t2]))

        self.features = None
        self.overlap_analysis = None
        self.tags_by_group = defaultdict(list)
        self.removed_ngrams = defaultdict(lambda: {"trigrams": [], "fourgrams": [], "unigrams": [], "bigrams": [], "fivegrams": []})
        self.kept_shared = defaultdict(lambda: {"trigrams": [], "fourgrams": [], "unigrams": [], "bigrams": [], "fivegrams": []})
        
        self.ngram_doc_count = defaultdict(int)
        self.total_tags = 0

    def load_data(self) -> None:
        print(f"Loading auto n-grams from: {self.auto_ngrams_path}")
        with open(self.auto_ngrams_path, 'r', encoding='utf-8') as f:
            self.features = json.load(f)

        print(f"Loading overlap analysis from: {self.overlap_analysis_path}")
        with open(self.overlap_analysis_path, 'r', encoding='utf-8') as f:
            self.overlap_analysis = json.load(f)

        for tag, tag_data in self.features["tags"].items():
            group = tag_data.get("group", "all_tags")
            self.tags_by_group[group].append(tag)
        
        self.total_tags = len(self.features["tags"])

        print(f"Loaded features for {self.total_tags} tags")
        print(f"Grouped into {len(self.tags_by_group)} groups")
        for group in sorted(self.tags_by_group.keys()):
            print(f"   {group}: {len(self.tags_by_group[group])} tags")
        
        if self.use_tfidf:
            self._compute_document_frequencies()

    def _compute_document_frequencies(self) -> None:
        print("Computing document frequencies for TF-IDF...")
        
        for tag, tag_data in self.features["tags"].items():
            for ngram_type in ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"]:
                ngrams = tag_data.get(ngram_type, [])
                for item in ngrams:
                    ngram = item["ngram"]
                    key = (ngram_type, ngram)
                    self.ngram_doc_count[key] += 1
        
        high_df = sum(1 for count in self.ngram_doc_count.values() if count > 10)
        print(f"   N-grams appearing in >10 tags: {high_df}")

    def compute_tfidf_weight(self, ngram: str, ngram_type: str, frequency: int) -> float:
        key = (ngram_type, ngram)
        doc_count = self.ngram_doc_count.get(key, 1)
        
        tf = math.log(1 + frequency)
        
        idf = math.log(self.total_tags / doc_count)
        
        tfidf = tf * idf
        max_tfidf = math.log(1 + 100) * math.log(self.total_tags)
        
        return min(1.0, tfidf / max_tfidf) if max_tfidf > 0 else 0.0

    def _are_confused_pair(self, tag1: str, tag2: str) -> bool:
        return frozenset([tag1, tag2]) in self.confused_pairs

    def check_dominance(
        self,
        ngram: str,
        tag: str,
        group: str,
        ngram_type: str
    ) -> bool:
        group_data = self.overlap_analysis["per_group"].get(group, {})
        
        overlap_key = f"{ngram_type[:-1]}_overlap"
        overlap_data = group_data.get(overlap_key, {})

        if ngram not in overlap_data:
            return True

        ngram_info = overlap_data[ngram]
        if ngram_info["overlap_count"] == 1:
            return True

        frequencies = ngram_info["frequencies"]
        tag_freq = frequencies.get(tag, 0)

        if tag_freq == 0:
            return False

        other_tags = [t for t in frequencies.keys() if t != tag]
        if not other_tags:
            return True

        other_freqs = [frequencies[t] for t in other_tags]
        max_other_freq = max(other_freqs)

        dominance_threshold = self.dominance_ratio
        
        if self.confused_pairs:
            for other_tag in other_tags:
                if self._are_confused_pair(tag, other_tag):
                    dominance_threshold = self.strict_dominance_ratio
                    break

        is_dominant = tag_freq >= (dominance_threshold * max_other_freq)

        return is_dominant

    def clean_group_ngrams(
        self,
        group: str,
        ngram_type: str
    ) -> Dict[str, Set[str]]:
        tags_in_group = self.tags_by_group[group]
        group_data = self.overlap_analysis["per_group"].get(group, {})
        overlap_key = f"{ngram_type[:-1]}_overlap"
        overlap_data = group_data.get(overlap_key, {})

        tag_ngrams_to_keep = {}
        for tag in tags_in_group:
            tag_data = self.features["tags"][tag]
            ngrams_list = tag_data.get(ngram_type, [])
            all_ngrams = {item["ngram"] for item in ngrams_list}
            tag_ngrams_to_keep[tag] = all_ngrams.copy()

        shared_ngrams = [
            ngram for ngram, info in overlap_data.items()
            if info["overlap_count"] > 1
        ]

        for ngram in shared_ngrams:
            ngram_info = overlap_data[ngram]
            tags_with_ngram = ngram_info["tags"]

            for tag in tags_with_ngram:
                if tag not in tags_in_group:
                    continue

                is_dominant = self.check_dominance(ngram, tag, group, ngram_type)

                if not is_dominant:
                    if ngram in tag_ngrams_to_keep[tag]:
                        tag_ngrams_to_keep[tag].remove(ngram)
                        self.removed_ngrams[tag][ngram_type].append({
                            "ngram": ngram,
                            "group": group,
                            "frequencies": ngram_info["frequencies"],
                            "reason": "not_dominant"
                        })
                else:
                    self.kept_shared[tag][ngram_type].append({
                        "ngram": ngram,
                        "group": group,
                        "frequencies": ngram_info["frequencies"]
                    })

        return tag_ngrams_to_keep

    def apply_safety_net(
        self,
        tag: str,
        ngrams_to_keep: Set[str],
        ngram_type: str
    ) -> Set[str]:
        if len(ngrams_to_keep) >= self.min_ngrams_per_tag:
            return ngrams_to_keep

        needed = self.min_ngrams_per_tag - len(ngrams_to_keep)
        tag_data = self.features["tags"][tag]
        original_ngrams = tag_data.get(ngram_type, [])

        removed = [
            item for item in original_ngrams
            if item["ngram"] not in ngrams_to_keep
        ]
        removed_sorted = sorted(removed, key=lambda x: x["rank"])

        restored = set()
        for item in removed_sorted[:needed]:
            ngrams_to_keep.add(item["ngram"])
            restored.add(item["ngram"])

        if restored:
            print(f"  Safety net: Restored {len(restored)} {ngram_type} for {tag}")

        return ngrams_to_keep

    def _apply_tfidf_reranking(self, tag_data: Dict, ngrams_to_keep: Set[str], ngram_type: str) -> List[Dict]:
        original_ngrams = tag_data.get(ngram_type, [])
        kept_ngrams = [item for item in original_ngrams if item["ngram"] in ngrams_to_keep]
        
        for item in kept_ngrams:
            item["tfidf_score"] = self.compute_tfidf_weight(
                item["ngram"], ngram_type, item["frequency"]
            )
        
        kept_ngrams.sort(key=lambda x: x["tfidf_score"], reverse=True)
        
        for i, item in enumerate(kept_ngrams):
            item["rank"] = i + 1
        
        return kept_ngrams

    def clean_all_groups(self) -> Dict:
        print("\n" + "="*70)
        print("CLEANING N-GRAMS (WITHIN-CLUSTER DOMINANCE-BASED)")
        print("="*70)
        print(f"Dominance ratio: {self.dominance_ratio}x")
        if self.confused_pairs:
            print(f"Strict dominance ratio: {self.strict_dominance_ratio}x (for {len(self.confused_pairs)} confused pairs)")
        if self.use_tfidf:
            print(f"TF-IDF weighting: Enabled")
        print(f"Min n-grams per tag: {self.min_ngrams_per_tag}")
        print("="*70)

        cleaned_features = {
            "metadata": self.features["metadata"].copy(),
            "tags": {}
        }

        cleaned_features["metadata"]["cleaned"] = True
        cleaned_features["metadata"]["dominance_ratio"] = self.dominance_ratio
        cleaned_features["metadata"]["min_ngrams_per_tag"] = self.min_ngrams_per_tag
        cleaned_features["metadata"]["cleanup_scope"] = "within_group"
        cleaned_features["metadata"]["use_tfidf"] = self.use_tfidf
        if self.confused_pairs:
            cleaned_features["metadata"]["strict_dominance_ratio"] = self.strict_dominance_ratio
            cleaned_features["metadata"]["num_confused_pairs"] = len(self.confused_pairs)

        ngram_types = ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"]

        for group in sorted(self.tags_by_group.keys()):
            print(f"\nCleaning group: {group}")
            tags_in_group = self.tags_by_group[group]
            print(f"   Tags: {len(tags_in_group)}")

            ngrams_to_keep = {}
            for ngram_type in ngram_types:
                ngrams_to_keep[ngram_type] = self.clean_group_ngrams(group, ngram_type)

            for tag in tags_in_group:
                tag_data = self.features["tags"][tag]
                cleaned_tag_data = {
                    "group": tag_data.get("group", "all_tags")
                }

                for ngram_type in ngram_types:
                    kept = self.apply_safety_net(
                        tag, ngrams_to_keep[ngram_type].get(tag, set()), ngram_type
                    )

                    if self.use_tfidf:
                        cleaned_ngrams = self._apply_tfidf_reranking(tag_data, kept, ngram_type)
                    else:
                        cleaned_ngrams = [
                            item for item in tag_data.get(ngram_type, [])
                            if item["ngram"] in kept
                        ]
                    
                    cleaned_tag_data[ngram_type] = cleaned_ngrams
                    cleaned_tag_data[f"total_{ngram_type}"] = tag_data.get(f"total_{ngram_type}", len(cleaned_ngrams))

                cleaned_features["tags"][tag] = cleaned_tag_data

                total_removed = sum(
                    len(tag_data.get(nt, [])) - len(cleaned_tag_data[nt])
                    for nt in ngram_types
                )
                if total_removed > 0:
                    print(f"   {tag}: removed {total_removed} n-grams total")

        print("\n" + "="*70)
        print("Cleanup complete!")
        print("="*70)

        return cleaned_features

    def generate_cleanup_report(self, cleaned_features: Dict) -> Dict:
        print("\nGenerating cleanup report...")

        ngram_types = ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"]

        original_counts = {nt: 0 for nt in ngram_types}
        cleaned_counts = {nt: 0 for nt in ngram_types}

        for tag_data in self.features["tags"].values():
            for nt in ngram_types:
                original_counts[nt] += len(tag_data.get(nt, []))

        for tag_data in cleaned_features["tags"].values():
            for nt in ngram_types:
                cleaned_counts[nt] += len(tag_data.get(nt, []))

        per_group_stats = {}
        for group in sorted(self.tags_by_group.keys()):
            tags_in_group = self.tags_by_group[group]
            group_stats = {"tags": len(tags_in_group)}

            for nt in ngram_types:
                group_orig = sum(
                    len(self.features["tags"][tag].get(nt, []))
                    for tag in tags_in_group
                )
                group_clean = sum(
                    len(cleaned_features["tags"][tag].get(nt, []))
                    for tag in tags_in_group
                )
                removed = group_orig - group_clean
                group_stats[nt] = {
                    "original": group_orig,
                    "cleaned": group_clean,
                    "removed": removed,
                    "removed_pct": (removed / group_orig * 100) if group_orig > 0 else 0
                }

            per_group_stats[group] = group_stats

        per_tag_details = {}
        for tag in sorted(self.features["tags"].keys()):
            tag_detail = {
                "group": self.features["tags"][tag].get("group", "all_tags")
            }
            for nt in ngram_types:
                orig = len(self.features["tags"][tag].get(nt, []))
                clean = len(cleaned_features["tags"][tag].get(nt, []))
                tag_detail[nt] = {
                    "original": orig,
                    "cleaned": clean,
                    "removed": len(self.removed_ngrams[tag].get(nt, [])),
                    "kept_shared": len(self.kept_shared[tag].get(nt, []))
                }
            tag_detail["removed_ngrams"] = dict(self.removed_ngrams[tag])
            tag_detail["kept_shared_ngrams"] = dict(self.kept_shared[tag])
            per_tag_details[tag] = tag_detail

        summary = {
            "total_tags": len(self.features["tags"]),
            "total_groups": len(self.tags_by_group)
        }
        for nt in ngram_types:
            removed = original_counts[nt] - cleaned_counts[nt]
            summary[nt] = {
                "original": original_counts[nt],
                "cleaned": cleaned_counts[nt],
                "removed": removed,
                "removed_pct": (removed / original_counts[nt] * 100) if original_counts[nt] > 0 else 0
            }

        report = {
            "metadata": {
                "dominance_ratio": self.dominance_ratio,
                "min_ngrams_per_tag": self.min_ngrams_per_tag,
                "cleanup_scope": "within_group",
                "use_tfidf": self.use_tfidf,
                "strict_dominance_ratio": self.strict_dominance_ratio if self.confused_pairs else None,
                "num_confused_pairs": len(self.confused_pairs) if self.confused_pairs else 0
            },
            "summary": summary,
            "per_group": per_group_stats,
            "per_tag": per_tag_details
        }

        return report

    def print_summary(self, report: Dict) -> None:
        print("\n" + "="*70)
        print("CLEANUP REPORT SUMMARY")
        print("="*70)

        summary = report["summary"]
        print(f"\nGLOBAL STATISTICS:")
        print(f"   Total tags: {summary['total_tags']}")
        print(f"   Total groups: {summary['total_groups']}")
        print(f"   Dominance ratio: {report['metadata']['dominance_ratio']}x")
        if report['metadata'].get('use_tfidf'):
            print(f"   TF-IDF weighting: Enabled")
        if report['metadata'].get('num_confused_pairs', 0) > 0:
            print(f"   Confused pairs: {report['metadata']['num_confused_pairs']} (strict ratio: {report['metadata']['strict_dominance_ratio']}x)")

        for nt in ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"]:
            if nt in summary:
                stats = summary[nt]
                print(f"\n   {nt.upper()}:")
                print(f"      Original: {stats['original']}")
                print(f"      Cleaned:  {stats['cleaned']}")
                print(f"      Removed:  {stats['removed']} ({stats['removed_pct']:.1f}%)")

        print("\n" + "="*70)

    def save_cleaned_features(self, cleaned_features: Dict, output_path: Path) -> None:
        print(f"\nSaving cleaned features to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_features, f, ensure_ascii=False, indent=2)
        print("Saved successfully!")

    def save_cleanup_report(self, report: Dict, output_path: Path) -> None:
        print(f"Saving cleanup report to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Saved successfully!")

    def clean_and_save(
        self,
        manual_ngrams_path: Path,
        cleanup_report_path: Path
    ) -> Tuple[Dict, Dict]:
        self.load_data()

        cleaned_features = self.clean_all_groups()

        report = self.generate_cleanup_report(cleaned_features)

        self.print_summary(report)

        self.save_cleaned_features(cleaned_features, manual_ngrams_path)
        self.save_cleanup_report(report, cleanup_report_path)

        return cleaned_features, report

def get_confused_pairs_from_analysis(overlap_analysis_path: Path, top_k: int = 10) -> List[Tuple[str, str]]:
    with open(overlap_analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    confused_pairs = []
    
    for group_name, group_data in analysis.get("per_group", {}).items():
        stats = group_data.get("statistics", {})
        top_overlaps = stats.get("top_tag_overlaps", [])
        
        for item in top_overlaps[:top_k]:
            tags = item.get("tags", [])
            if len(tags) == 2:
                confused_pairs.append((tags[0], tags[1]))
    
    return confused_pairs[:top_k]
