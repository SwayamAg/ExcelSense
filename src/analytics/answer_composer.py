"""
answer_composer.py
==================
Turns executed analytical steps into a structured business answer.

Every sentence produced here is generated from numbers that
`analytics_core` already computed. The composer never estimates, never
extrapolates and never asserts causation - it converts a StepResult list into:

    Answer               one-line executive conclusion
    Evidence             the key numbers, with denominators
    Analysis             what was compared and how
    Business implication why it matters, in portfolio terms
    Recommendation       what management could do or investigate next
    Caveat               correlation-vs-causation and sample-size warnings

An optional LLM pass (`polish`) may rewrite the prose, but it is handed only
the already-computed figures and is explicitly forbidden from adding numbers.
The deterministic answer is always kept and always shown.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import analytics_core as ac
from analysis_planner import AnalyticalPlan, StepResult

# ---------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------

def fmt_metric(metric: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    spec = ac.METRICS.get(metric)
    if spec is None:
        if metric in ac.SHARE_METRICS:
            return f"{float(value):,.2f}%"
        return f"{value}"
    return spec.format(float(value))


def fmt_inr(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    v = float(value)
    if abs(v) >= 1e7:
        return f"Rs {v / 1e7:,.2f} cr"
    if abs(v) >= 1e5:
        return f"Rs {v / 1e5:,.2f} L"
    return f"Rs {v:,.0f}"


def metric_label(metric: str) -> str:
    spec = ac.METRICS.get(metric)
    if spec:
        return spec.label
    # SHARE_METRICS is a list of metric names, not a dict – just humanise the name
    return metric.replace("_", " ")


def gap_phrase(metric: str, diff: float) -> str:
    """'4.5 percentage points above' / '0.32x below' - unit-aware."""
    spec = ac.METRICS.get(metric)
    unit = spec.unit if spec else "%"
    direction = "above" if diff > 0 else "below"
    if unit == "%":
        return f"{abs(diff):.1f} percentage points {direction}"
    if unit == "x":
        return f"{abs(diff):.2f}x {direction}"
    if unit == "INR":
        return f"{fmt_inr(abs(diff))} {direction}"
    return f"{abs(diff):,.2f} {direction}"


def reliable_rows(df: pd.DataFrame, metric: Optional[str] = None) -> pd.DataFrame:
    """Rows whose sample size cleared the threshold, falling back to all rows
    when nothing qualifies (so an answer is still produced, with a caveat)."""
    if df is None or df.empty:
        return df
    col = None
    if metric and f"{metric}_reliable" in df.columns:
        col = f"{metric}_reliable"
    elif "reliable" in df.columns:
        col = "reliable"
    if col is None:
        return df
    kept = df[df[col].fillna(False).astype(bool)]
    return kept if not kept.empty else df


def headline_metric(df: pd.DataFrame, exclude_counts: bool = True) -> Optional[str]:
    """The metric column a reader should be shown first.

    Rate and ratio metrics beat volume metrics: a retention table whose columns
    are `retention_contacts` and `retention_success_rate` must lead with the
    success rate, not with "SMS at 16".
    """
    cols = [c for c in df.columns if c in ac.METRICS or c in ac.SHARE_METRICS]
    if not cols:
        return None
    rates = [c for c in cols if c in ac.METRICS and ac.METRICS[c].unit in ("%", "x")]
    if rates:
        return rates[0]
    if exclude_counts:
        non_count = [c for c in cols
                     if c in ac.SHARE_METRICS
                     or (c in ac.METRICS and ac.METRICS[c].unit != "count")]
        if non_count:
            return non_count[0]
    return cols[0]


def adverse_driver(ds_df: pd.DataFrame) -> Optional[pd.Series]:
    """From a driver_scan result, the largest deviation in the *bad* direction.

    Sorting by absolute deviation puts the most striking segment first, which is
    right for evidence but wrong for a recommendation: a segment performing far
    BETTER than average is not somewhere to intervene.
    """
    if ds_df is None or ds_df.empty:
        return None
    worse = ds_df[ds_df["direction"].astype(str).str.startswith("worse")]
    if not worse.empty:
        return worse.iloc[0]
    return None


def dim_column_of(df: pd.DataFrame) -> Optional[str]:
    """First non-metric column of a metric table - i.e. the grouping key."""
    for c in df.columns:
        if c in ("rank", "baseline", "diff_vs_baseline", "lift", "reliable",
                 "priority_score", "n", "deviation", "is_outlier", "method"):
            continue
        if c.endswith("_n") or c.endswith("_reliable") or c.startswith("score_"):
            continue
        if c in ac.METRICS or c in ac.SHARE_METRICS:
            continue
        return c
    return df.columns[0] if len(df.columns) else None


# ---------------------------------------------------------------------
# answer object
# ---------------------------------------------------------------------

@dataclass
class BusinessAnswer:
    question: str
    headline: str
    evidence: List[str] = field(default_factory=list)
    analysis: List[str] = field(default_factory=list)
    implication: List[str] = field(default_factory=list)
    recommendation: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    tables: List[Tuple[str, pd.DataFrame]] = field(default_factory=list)
    plan_description: List[Dict[str, str]] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    metric_definitions: Dict[str, str] = field(default_factory=dict)
    llm_narrative: Optional[str] = None

    def to_markdown(self, include_tables: bool = True, max_rows: int = 10) -> str:
        out: List[str] = [f"### {self.headline}", ""]
        if self.evidence:
            out.append("**Evidence**")
            out.extend(f"- {e}" for e in self.evidence)
            out.append("")
        if self.analysis:
            out.append("**Analysis**")
            out.extend(f"- {a}" for a in self.analysis)
            out.append("")
        if self.implication:
            out.append("**Business implication**")
            out.extend(f"- {i}" for i in self.implication)
            out.append("")
        if self.recommendation:
            out.append("**Recommendation**")
            out.extend(f"- {r}" for r in self.recommendation)
            out.append("")
        if self.caveats:
            out.append("**Caveat**")
            out.extend(f"- {c}" for c in self.caveats)
            out.append("")
        if self.unavailable:
            out.append("**Not answerable from this dataset**")
            out.extend(f"- {u}" for u in self.unavailable)
            out.append("")
        if include_tables and self.tables:
            out.append("**Supporting tables**")
            for title, df in self.tables:
                out.append(f"\n*{title}*\n")
                out.append(df.head(max_rows).to_markdown(index=False))
            out.append("")
        return "\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.headline,
            "evidence": self.evidence,
            "analysis": self.analysis,
            "business_implication": self.implication,
            "recommendation": self.recommendation,
            "caveat": self.caveats,
            "unavailable": self.unavailable,
            "metric_definitions": self.metric_definitions,
            "analytical_plan": self.plan_description,
            "tables": {t: df.to_dict("records") for t, df in self.tables},
        }


# ---------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------

class AnswerComposer:
    """Deterministic narrative builder, dispatched by answer_template."""

    def compose(self, plan: AnalyticalPlan, results: Sequence[StepResult]) -> BusinessAnswer:
        ans = BusinessAnswer(question=plan.question, headline="")
        ans.plan_description = plan.describe()

        ok = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]
        for r in failed:
            ans.unavailable.append(f"{r.step.title}: {r.error}")

        # collect every table for the UI, and every metric definition touched
        for r in ok:
            if r.is_frame and not r.result.empty:
                ans.tables.append((r.step.title, r.result))
        for m in self._metrics_touched(results):
            spec = ac.METRICS.get(m)
            if spec:
                ans.metric_definitions[spec.label] = spec.definition

        handler = getattr(self, f"_t_{plan.answer_template}", None) or self._t_leaderboard
        try:
            handler(plan, ok, ans)
        except Exception as e:  # never let narrative generation kill the answer
            ans.headline = ans.headline or "Analysis completed; see the supporting tables."
            ans.analysis.append(f"(Narrative generation issue: {type(e).__name__}: {e})")

        if not ans.headline:
            ans.headline = "The analysis ran but produced no ranked result."
        if plan.caveat:
            ans.caveats.insert(0, plan.caveat)
        self._add_reliability_caveats(ok, ans)
        return ans

    # ---- shared utilities -------------------------------------------
    @staticmethod
    def _metrics_touched(results: Sequence[StepResult]) -> List[str]:
        out: List[str] = []
        for r in results:
            p = r.step.params
            for m in ([p.get("metric")] + list(p.get("metrics") or []) +
                      [c.get("metric") for c in (p.get("components") or [])]):
                if isinstance(m, str) and m not in out:
                    out.append(m)
        return out

    @staticmethod
    def _find(results: Sequence[StepResult], op: str,
              nth: int = 0) -> Optional[StepResult]:
        hits = [r for r in results if r.step.op == op]
        return hits[nth] if len(hits) > nth else None

    @staticmethod
    def _find_all(results: Sequence[StepResult], op: str) -> List[StepResult]:
        return [r for r in results if r.step.op == op]

    @staticmethod
    def _overview(results: Sequence[StepResult]) -> Optional[Dict[str, Any]]:
        r = AnswerComposer._find(results, "portfolio_overview")
        return r.result if r else None

    def _add_reliability_caveats(self, results: Sequence[StepResult], ans: BusinessAnswer):
        thin = set()
        for r in results:
            if not r.is_frame or r.result.empty:
                continue
            df = r.result
            for col in df.columns:
                if col.endswith("_reliable"):
                    metric = col[:-len("_reliable")]
                    if (~df[col].fillna(False).astype(bool)).any():
                        thin.add(metric_label(metric))
            if "reliable" in df.columns and (~df["reliable"].fillna(False).astype(bool)).any():
                m = r.step.params.get("metric")
                thin.add(metric_label(m) if m else r.step.title)
        if thin:
            ans.caveats.append(
                "Some groups fell below the minimum sample size for "
                + ", ".join(sorted(thin))
                + "; those rows are flagged `reliable = False` and were not used "
                  "to draw the conclusion.")

    @staticmethod
    def _kpi_line(ov: Dict[str, Any], key: str) -> Optional[str]:
        d = ov.get(key)
        if not d or d.get("value") is None:
            return None
        if isinstance(d["value"], float) and math.isnan(d["value"]):
            return None

        # `portfolio_overview` emits the rendered figure as `display`. Reading a
        # non-existent `formatted` key printed "Total policies: None" beside a
        # headline that had the number, which is the worst of both.
        shown = d.get("display") or d.get("formatted")
        if shown is None:
            spec_fmt = ac.METRICS.get(key)
            shown = spec_fmt.format(d["value"]) if spec_fmt else f"{d['value']}"
        # "25,000 count" reads as a typo; a count needs no unit word.
        if d.get("unit") == "count":
            shown = str(shown).replace(" count", "")

        spec = ac.METRICS.get(key)
        base = f"**{d.get('label', key)}**: {shown}"
        n = d.get("denominator")
        # A plain count has denominator 1, so "(denominator: 1 policies)" is noise.
        if spec and spec.denominator_is_count and n and float(n) > 1:
            base += f" (of {n:,.0f})"
        return base

    # =================================================================
    # TEMPLATES
    # =================================================================

    # ---- A. full executive assessment --------------------------------
    def _t_executive_assessment(self, plan, results, ans):
        ov = self._overview(results)
        if not ov:
            return self._t_leaderboard(plan, results, ans)

        pc = ov["policy_count"]["value"]
        pers = ov.get("persistency_rate", {}).get("value")
        lapse = ov.get("lapse_rate", {}).get("value")
        lr = ov.get("loss_ratio", {}).get("value")
        rlr = ov.get("risk_loss_ratio", {}).get("value")
        sa = ov.get("total_sum_assured", {}).get("value")

        verdict = "broadly stable"
        if lapse is not None and lapse >= 25:
            verdict = "under persistency pressure"
        elif lapse is not None and lapse >= 20:
            verdict = "showing meaningful persistency strain"
        ans.headline = (f"The portfolio is {verdict}: {pc:,.0f} policies and "
                        f"{fmt_inr(sa)} of sum assured, with persistency at "
                        f"{fmt_metric('persistency_rate', pers)} and lapse at "
                        f"{fmt_metric('lapse_rate', lapse)}.")

        for key in ["policy_count", "customer_count", "total_sum_assured",
                    "total_annual_premium", "persistency_rate", "lapse_rate",
                    "surrender_rate", "claim_rate", "loss_ratio", "risk_loss_ratio",
                    "retention_success_rate"]:
            line = self._kpi_line(ov, key)
            if line:
                ans.evidence.append(line)
        mix = ov.get("_status_mix") or {}
        if mix:
            ans.evidence.append("**Status mix**: " + ", ".join(
                f"{k} {v:.1f}%" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))

        # product / channel / zone highlights
        for res in self._find_all(results, "multi_metric_table"):
            df = res.result
            dcol = dim_column_of(df)
            if dcol is None or "lapse_rate" not in df.columns:
                continue
            d = df.dropna(subset=["lapse_rate"])
            if d.empty:
                continue
            worst = d.sort_values("lapse_rate", ascending=False).iloc[0]
            best = d.sort_values("lapse_rate", ascending=True).iloc[0]
            ans.analysis.append(
                f"By {dcol.replace('_', ' ')}: worst persistency is **{worst[dcol]}** "
                f"({fmt_metric('lapse_rate', worst['lapse_rate'])} lapse, "
                f"n={worst.get('lapse_rate_n', float('nan')):,.0f}); best is "
                f"**{best[dcol]}** ({fmt_metric('lapse_rate', best['lapse_rate'])}).")

        conc = self._find(results, "concentration")
        if conc and isinstance(conc.result, dict):
            c = conc.result
            ans.evidence.append(
                f"**Concentration by {c['dimension']}**: HHI {c['hhi']} "
                f"({c['interpretation']}); top {c['top_n']} hold "
                f"{c['top_n_share_pct']:.1f}% of {metric_label(c['value_metric']).lower()}.")

        cmp_res = self._find(results, "compare_to_baseline")
        if cmp_res and cmp_res.is_frame and not cmp_res.result.empty:
            m = cmp_res.step.params["metric"]
            df = cmp_res.result
            dcol = dim_column_of(df)
            top = df.iloc[0]
            ans.analysis.append(
                f"{metric_label(m)} by {dcol.replace('_', ' ')} is led by "
                f"**{top[dcol]}** at {fmt_metric(m, top[m])}, "
                f"{gap_phrase(m, top['diff_vs_baseline'])} the portfolio baseline of "
                f"{fmt_metric(m, top['baseline'])}.")

        if lapse is not None:
            ans.implication.append(
                f"At a {fmt_metric('lapse_rate', lapse)} lapse rate, roughly "
                f"{ov['lapse_rate']['denominator'] * lapse / 100:,.0f} of "
                f"{ov['lapse_rate']['denominator']:,.0f} exposed policies have already "
                f"stopped paying - that is the size of the persistency problem.")
        if lr is not None and rlr is not None:
            ans.implication.append(
                f"Overall claims outflow is {fmt_metric('loss_ratio', lr)} of premium "
                f"collected, but only {fmt_metric('risk_loss_ratio', rlr)} is "
                f"underwriting loss; the rest is contractual maturity and surrender "
                f"payout, which is expected rather than adverse.")

        ans.recommendation.append(
            "Treat persistency as the primary agenda item: the products and channels "
            "listed above carry lapse rates materially off the book average.")
        ans.recommendation.append(
            "Read the risk-only loss ratio, not the headline loss ratio, when judging "
            "product economics - maturity-heavy products look worse than they are.")
        ans.caveats.append(
            "All figures are the current state of the book. The dataset holds no "
            "expense, commission or reserving data, so nothing here is a profitability "
            "statement.")

    def _t_kpi_block(self, plan, results, ans):
        ov = self._overview(results)
        if not ov:
            return self._t_leaderboard(plan, results, ans)
        ans.headline = "Portfolio KPI block, each metric with the denominator it was computed on."
        for key, d in ov.items():
            if key.startswith("_"):
                continue
            line = self._kpi_line(ov, key)
            if line:
                ans.evidence.append(line)
        mix = ov.get("_status_counts") or {}
        if mix:
            ans.analysis.append("Policy status counts: " + ", ".join(
                f"{k} {v:,}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))
        ans.analysis.append(
            "Denominators differ deliberately: lapse and persistency exclude Matured "
            "and Death-Claim policies (they ended by design), and surrender rate is "
            "measured only over products that carry a surrender value.")
        ans.implication.append(
            "These are the numbers every other answer in this system is benchmarked "
            "against, so a segment is only 'high risk' if it sits materially off one "
            "of these baselines.")

    def _t_risk_register(self, plan, results, ans):
        ov = self._overview(results)
        risks: List[Tuple[float, str]] = []

        for res in self._find_all(results, "compare_to_baseline"):
            df, m = res.result, res.step.params["metric"]
            if df.empty:
                continue
            rel = df[df.get("reliable", True).astype(bool)] if "reliable" in df else df
            if rel.empty:
                rel = df
            dcol = dim_column_of(rel)
            spec = ac.METRICS.get(m)
            # Find the most adverse row (highest if high is bad, lowest if high is good)
            if spec and spec.higher_is_better is False:
                adv = rel.sort_values(by="diff_vs_baseline", ascending=False)
            else:
                adv = rel.sort_values(by="diff_vs_baseline", ascending=True)
            if adv.empty:
                continue
            top = adv.iloc[0]
            diff = float(top.get("diff_vs_baseline") or 0)
            is_risk = (diff > 0) if (spec and spec.higher_is_better is False) else (diff < 0)
            if not is_risk and abs(diff) < 1e-4:
                continue
            base_val = abs(float(top.get("baseline", 1.0))) or 1e-9
            risks.append((abs(diff / base_val),
                          f"**{top[dcol]}** has a {metric_label(m).lower()} of "
                          f"{fmt_metric(m, top[m])}, {gap_phrase(m, diff)} the "
                          f"portfolio baseline of {fmt_metric(m, top['baseline'])} "
                          f"(n={top.get(f'{m}_n', top.get('sample_size', float('nan'))):,.0f})."))

        conc = self._find(results, "concentration")
        if conc and isinstance(conc.result, dict):
            c = conc.result
            if c["hhi"] > 0.15:
                risks.append((c["hhi"],
                              f"**Concentration risk**: {metric_label(c['value_metric']).lower()} "
                              f"is {c['interpretation']} by {c['dimension']} "
                              f"(HHI {c['hhi']}, effective spread {c['effective_n']} groups); "
                              f"the top {c['top_n']} hold {c['top_n_share_pct']:.1f}%."))

        tec = self._find(results, "top_entity_concentration")
        if tec and isinstance(tec.result, dict):
            t = tec.result
            risks.append((t["top_n_share_pct"] / 100.0,
                          f"**Single-policy exposure**: the largest {t['top_n']} of "
                          f"{t['n_records']:,} records carry {t['top_n_share_pct']:.1f}% "
                          f"of total {t['column'].replace('_', ' ')} ({fmt_inr(t['top_n_value'])})."))

        out = self._find(results, "outliers")
        if out and out.is_frame and not out.result.empty:
            flagged = out.result[out.result["is_outlier"].fillna(False)]
            m = out.step.params["metric"]
            dcol = dim_column_of(out.result)
            if not flagged.empty:
                names = ", ".join(str(v) for v in flagged[dcol].head(5).tolist())
                risks.append((0.5, f"**Outlier {ac.DIMENSIONS[out.step.params['dimension']].label.lower()}s**: "
                                   f"{len(flagged)} flagged on {metric_label(m).lower()} "
                                   f"({out.result['method'].iloc[0]}) - {names}."))

        risks.sort(key=lambda x: -x[0])
        selected_risks = [t for _, t in risks[:6]]
        if selected_risks:
            n_risks = len(selected_risks)
            ans.headline = (f"The {n_risks} largest exposure{'s' if n_risks != 1 else ''} in the current "
                            f"book, ranked by how far each sits from the portfolio baseline.")
            ans.evidence.extend(selected_risks)
        else:
            ans.headline = "No segment sits materially off the portfolio baseline on the metrics tested."

        if ov:
            ans.analysis.append(
                f"Baselines used: lapse {fmt_metric('lapse_rate', ov.get('lapse_rate', {}).get('value'))}, "
                f"surrender {fmt_metric('surrender_rate', ov.get('surrender_rate', {}).get('value'))}, "
                f"loss ratio {fmt_metric('loss_ratio', ov.get('loss_ratio', {}).get('value'))}, "
                f"risk loss ratio {fmt_metric('risk_loss_ratio', ov.get('risk_loss_ratio', {}).get('value'))}.")
        ans.analysis.append(
            "Each risk was tested by computing the metric for every group, comparing "
            "it with the portfolio baseline, and keeping only groups whose sample size "
            "clears the reliability threshold.")
        ans.implication.append(
            "Concentration risks and persistency risks are different problems: the "
            "first is about how much of the book depends on one thing, the second is "
            "about whether the book stays on the books.")
        ans.recommendation.append(
            "Work the list top-down. Start with exposures that combine a large gap to "
            "baseline with a large share of the portfolio, because those are the only "
            "ones where a fix moves the aggregate number.")
        ans.caveats.append(
            "These are observed positions in the current book, not forecasts. Nothing "
            "here identifies why a segment underperforms.")

    # ---- ranked management agendas ------------------------------------
    def _t_priority_ranking(self, plan, results, ans):
        comps = self._find_all(results, "composite_score")
        if not comps:
            return self._t_leaderboard(plan, results, ans)

        first = comps[0]
        df = first.result
        dcol = dim_column_of(df)
        top = df.iloc[0]
        components = first.step.params["components"]
        frame_txt = ", ".join(
            f"{metric_label(c['metric']).lower()} ({int(float(c.get('weight', 1)) * 100)}%, "
            f"{'higher' if str(c.get('direction', 'high')).startswith('high') else 'lower'} "
            f"raises score)" for c in components)

        q_lower = (plan.question or "").lower()
        is_balance_q = any(w in q_lower for w in ("balance", "trade-off", "tradeoff", "best channel", "best product", "overall"))

        if is_balance_q:
            ans.headline = (f"**{top[dcol]}** ranks highest with a balanced multi-metric score of "
                            f"{top['priority_score']:.3f}, evaluating both growth and persistency.")
        else:
            ans.headline = (f"**{top[dcol]}** ranks first for attention, scoring "
                            f"{top['priority_score']:.3f} on a framework weighting "
                            f"{frame_txt}.")

        for _, row in df.head(4).iterrows():
            bits = []
            for c in components:
                m = c["metric"]
                if m in row and pd.notna(row[m]):
                    bits.append(f"{metric_label(m).lower()} {fmt_metric(m, row[m])}")
            ans.evidence.append(
                f"**{row[dcol]}** (rank {int(row['rank'])}, balanced score "
                f"{row['priority_score']:.3f}): " + "; ".join(bits))

        ans.analysis.append(
            f"The balance score integrates both growth metrics ({', '.join(metric_label(c['metric']).lower() for c in components if 'share' in c['metric'] or 'premium' in c['metric'] or 'count' in c['metric'])}) "
            f"and persistency/retention metrics ({', '.join(metric_label(c['metric']).lower() for c in components if 'persistency' in c['metric'] or 'lapse' in c['metric'] or 'retention' in c['metric'])}). "
            f"Each component was normalised and weighted according to the declared framework.")
        for res in comps[1:]:
            d2 = res.result
            if d2.empty:
                continue
            c2 = dim_column_of(d2)
            ans.analysis.append(
                f"The same framework applied to {ac.DIMENSIONS[res.step.params['dimension']].label.lower()} "
                f"puts **{d2.iloc[0][c2]}** first (score {d2.iloc[0]['priority_score']:.3f}).")

        for res in self._find_all(results, "compare_to_baseline"):
            d3, m = res.result, res.step.params["metric"]
            if d3.empty:
                continue
            c3 = dim_column_of(d3)
            r0 = d3.iloc[0]
            ans.evidence.append(
                f"Against the portfolio baseline, **{r0[c3]}** has the highest "
                f"{metric_label(m).lower()} at {fmt_metric(m, r0[m])}, "
                f"{gap_phrase(m, r0['diff_vs_baseline'])} the book average of "
                f"{fmt_metric(m, r0['baseline'])}.")

        share_cols = [c for c in ("sum_assured_share", "policy_share", "premium_share")
                      if c in df.columns]
        if share_cols and pd.notna(top.get(share_cols[0])):
            share = float(top[share_cols[0]])
            biggest = df.sort_values(share_cols[0], ascending=False).iloc[0]
            if share >= 15:
                ans.implication.append(
                    f"**{top[dcol]}** carries {share:.1f}% of the portfolio on "
                    f"{metric_label(share_cols[0]).lower()}, so fixing it moves the "
                    f"aggregate number rather than a corner of the book.")
            else:
                ans.implication.append(
                    f"**{top[dcol]}** ranks first on severity but is only {share:.1f}% "
                    f"of the portfolio on {metric_label(share_cols[0]).lower()}, so the "
                    f"absolute number of policies recovered will be small. "
                    f"**{biggest[dcol]}** is the largest block at "
                    f"{float(biggest[share_cols[0]]):.1f}% and is where the same "
                    f"percentage improvement would be worth the most.")
        else:
            ans.implication.append(
                f"**{top[dcol]}** tops the ranking because it combines poor performance "
                f"with enough scale to matter.")

        ds = self._find(results, "driver_scan")
        if ds and ds.is_frame and not ds.result.empty:
            r0 = adverse_driver(ds.result)
            m_ds = ds.step.params["metric"]
            if r0 is not None:
                ans.recommendation.append(
                    f"Target the intervention rather than the whole group: within "
                    f"**{top[dcol]}**, the worst segment is "
                    f"{r0['dimension_label']} = **{r0['segment']}** at "
                    f"{fmt_metric(m_ds, r0['value'])} "
                    f"({gap_phrase(m_ds, r0['diff_vs_baseline'])} that group's own "
                    f"average, n={r0['n']}).")
            else:
                ans.recommendation.append(
                    f"No sub-segment inside **{top[dcol]}** performs materially worse "
                    f"than the group as a whole, so the problem is spread across the "
                    f"group rather than concentrated - a broad intervention is the "
                    f"only option the data supports.")
        ans.recommendation.append(
            f"Re-run this ranking with different weights if management's priorities "
            f"differ - the framework is explicit, so the ranking can be argued with.")

    def _t_retention_priority(self, plan, results, ans):
        self._t_priority_ranking(plan, results, ans)
        ret = [r for r in self._find_all(results, "multi_metric_table")
               if r.is_frame and "retention_success_rate" in r.result.columns]
        if ret:
            df = reliable_rows(ret[0].result, "retention_success_rate").dropna(
                subset=["retention_success_rate"])
            if not df.empty:
                dcol = dim_column_of(df)
                worst = df.sort_values("retention_success_rate").iloc[0]
                best = df.sort_values("retention_success_rate", ascending=False).iloc[0]
                ans.evidence.append(
                    f"Retention effort currently saves "
                    f"{fmt_metric('retention_success_rate', best['retention_success_rate'])} "
                    f"of contacted **{best[dcol]}** cases but only "
                    f"{fmt_metric('retention_success_rate', worst['retention_success_rate'])} "
                    f"of **{worst[dcol]}** cases.")
                ans.implication.append(
                    "A high lapse rate and a low save rate together mean effort is "
                    "being spent without recovering the policy - that is where a "
                    "change of approach, not more calls, is needed.")

    def _t_strength_weakness(self, plan, results, ans):
        comp = self._find(results, "composite_score")
        if not comp:
            return self._t_leaderboard(plan, results, ans)
        df = comp.result
        dcol = dim_column_of(df)
        best, worst = df.iloc[0], df.iloc[-1]
        components = comp.step.params["components"]
        ans.headline = (f"**{best[dcol]}** is the strongest part of the book and "
                        f"**{worst[dcol]}** the weakest, on a scorecard weighting "
                        + ", ".join(f"{metric_label(c['metric']).lower()} "
                                    f"({int(float(c.get('weight', 1)) * 100)}%)"
                                    for c in components) + ".")
        for label, row in (("Strongest", best), ("Weakest", worst)):
            bits = [f"{metric_label(c['metric']).lower()} {fmt_metric(c['metric'], row.get(c['metric']))}"
                    for c in components if c["metric"] in row]
            ans.evidence.append(f"**{label} - {row[dcol]}** (score "
                                f"{row['priority_score']:.3f}): " + "; ".join(bits))
        ans.evidence.append("Full ranking: " + " > ".join(
            f"{r[dcol]} ({r['priority_score']:.2f})" for _, r in df.iterrows()))
        ans.analysis.append(
            "The score orients every component so that higher always means better, "
            "min-max normalises across groups with reliable denominators, then weights. "
            "The `score_*` columns show each component's contribution.")
        ov = self._overview(results)
        if ov:
            ans.implication.append(
                f"The gap matters in context: the portfolio lapse rate is "
                f"{fmt_metric('lapse_rate', ov.get('lapse_rate', {}).get('value'))} and "
                f"persistency {fmt_metric('persistency_rate', ov.get('persistency_rate', {}).get('value'))}, "
                f"so groups either side of that line pull the aggregate in opposite directions.")
        ans.recommendation.append(
            f"Study **{best[dcol]}** for what works - customer mix, billing pattern and "
            f"underwriting intensity are all in the supporting table - before designing "
            f"a fix for **{worst[dcol]}**.")

    # ---- rankings and leaderboards ------------------------------------
    def _t_leaderboard(self, plan, results, ans):
        cmp_res = self._find(results, "compare_to_baseline")
        table_res = [r for r in self._find_all(results, "multi_metric_table") if r.is_frame]

        primary = cmp_res or (table_res[0] if table_res else None)
        if primary is None:
            ov = self._overview(results)
            if ov:
                return self._t_kpi_block(plan, results, ans)
            ans.headline = "No ranked result was produced for this question."
            return

        df = primary.result
        dcol = dim_column_of(df)
        q_lower = (plan.question or "").lower()
        wants_lowest = bool(re.search(r"\b(lowest|worst|bottom|least|poor|poorest|underperforming|lagging|minimum|min)\b", q_lower))

        if primary.step.op == "compare_to_baseline":
            m = primary.step.params["metric"]
            rel = df[df["reliable"].fillna(False).astype(bool)] if "reliable" in df else df
            rel = rel if not rel.empty else df

            sorted_df = df.sort_values(by=m, ascending=True)
            sorted_rel = rel.sort_values(by=m, ascending=True)

            obs_lowest = sorted_df.iloc[0]
            rel_lowest = sorted_rel.iloc[0]

            if wants_lowest:
                obs_n = obs_lowest.get(f'{m}_n', obs_lowest.get('sample_size', float('nan')))
                rel_n = rel_lowest.get(f'{m}_n', rel_lowest.get('sample_size', float('nan')))
                if str(obs_lowest[dcol]) != str(rel_lowest[dcol]) and pd.notna(obs_n) and obs_n < 20:
                    ans.headline = (f"**{obs_lowest[dcol]}** has the lowest observed {metric_label(m).lower()} at "
                                    f"{fmt_metric(m, obs_lowest[m])} (n={obs_n:,.0f}, below sample threshold); "
                                    f"among reliable segments (n >= 30), **{rel_lowest[dcol]}** has the lowest at "
                                    f"{fmt_metric(m, rel_lowest[m])} ({gap_phrase(m, rel_lowest['diff_vs_baseline'])} baseline).")
                else:
                    ans.headline = (f"**{rel_lowest[dcol]}** has the lowest {metric_label(m).lower()} at "
                                    f"{fmt_metric(m, rel_lowest[m])}, {gap_phrase(m, rel_lowest['diff_vs_baseline'])} "
                                    f"the portfolio baseline of {fmt_metric(m, rel_lowest['baseline'])} (n={rel_n:,.0f}).")
                for _, row in sorted_rel.head(5).iterrows():
                    ans.evidence.append(
                        f"**{row[dcol]}**: {fmt_metric(m, row[m])} "
                        f"({gap_phrase(m, row['diff_vs_baseline'])} baseline, "
                        f"n={row.get(f'{m}_n', row.get('sample_size', float('nan'))):,.0f})")
                if str(obs_lowest[dcol]) != str(rel_lowest[dcol]):
                    ans.evidence.append(
                        f"⚠️ **{obs_lowest[dcol]}** (small sample caution): {fmt_metric(m, obs_lowest[m])} "
                        f"(n={obs_n:,.0f}, below reliability threshold).")
            else:
                top, bottom = rel.iloc[0], rel.iloc[-1]
                ans.headline = (f"**{top[dcol]}** leads on {metric_label(m).lower()} at "
                                f"{fmt_metric(m, top[m])}, {gap_phrase(m, top['diff_vs_baseline'])} "
                                f"the portfolio baseline of {fmt_metric(m, top['baseline'])}.")
                for _, row in rel.head(5).iterrows():
                    ans.evidence.append(
                        f"**{row[dcol]}**: {fmt_metric(m, row[m])} "
                        f"({gap_phrase(m, row['diff_vs_baseline'])} baseline, "
                        f"n={row.get(f'{m}_n', float('nan')):,.0f})")
                if len(rel) > 1:
                    ans.evidence.append(
                        f"Lowest: **{bottom[dcol]}** at {fmt_metric(m, bottom[m])} "
                        f"({gap_phrase(m, bottom['diff_vs_baseline'])} baseline).")
            spec = ac.METRICS.get(m)
            if spec:
                ans.analysis.append(f"{spec.label} is defined as: {spec.definition}")
            ans.analysis.append(
                f"The spread between best and worst is "
                f"{gap_phrase(m, float(rel.iloc[0][m]) - float(rel.iloc[-1][m])).replace(' above', '').replace(' below', '')}.")
        else:
            metrics = [c for c in df.columns if c in ac.METRICS or c in ac.SHARE_METRICS]
            lead = metrics[0] if metrics else None
            if lead is None:
                ans.headline = primary.step.title
            else:
                top = df.iloc[0]
                ans.headline = (f"On {metric_label(lead).lower()}, **{top[dcol]}** leads at "
                                f"{fmt_metric(lead, top[lead])} across {len(df)} "
                                f"{dcol.replace('_', ' ')} groups.")
                for _, row in df.head(5).iterrows():
                    bits = [f"{metric_label(m).lower()} {fmt_metric(m, row[m])}"
                            for m in metrics[:5] if pd.notna(row.get(m))]
                    ans.evidence.append(f"**{row[dcol]}**: " + "; ".join(bits))
                ans.analysis.append(
                    f"{len(metrics)} metrics were computed for each of {len(df)} groups: "
                    + ", ".join(metric_label(m).lower() for m in metrics) + ".")

        for res in table_res:
            if res is primary:
                continue
            d2 = res.result
            c2 = dim_column_of(d2)
            m2 = [c for c in d2.columns if c in ac.METRICS or c in ac.SHARE_METRICS]
            if not m2 or d2.empty or c2 is None:
                continue
            r0 = d2.iloc[0]
            ans.analysis.append(
                f"{res.step.title}: **{r0[c2]}** leads on {metric_label(m2[0]).lower()} "
                f"at {fmt_metric(m2[0], r0[m2[0]])}.")

        conc = self._find(results, "concentration")
        if conc and isinstance(conc.result, dict):
            c = conc.result
            ans.implication.append(
                f"{metric_label(c['value_metric'])} is {c['interpretation']} across "
                f"{c['dimension']} (HHI {c['hhi']}); the top {c['top_n']} account for "
                f"{c['top_n_share_pct']:.1f}%.")
        if not ans.implication:
            ans.implication.append(
                "Differences of this size are worth acting on only where the group is "
                "big enough to move the portfolio number - the `_n` columns show how "
                "many records sit behind each row.")
        ans.recommendation.append(
            "Confirm the leading group's result holds once volume is taken into "
            "account, then decompose it before designing an intervention.")

    def _t_risk_segmentation(self, plan, results, ans):
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        ds = self._find(results, "driver_scan")

        rows: List[Tuple[float, str, Any, Any, Any, Any]] = []
        for res in cmps:
            m = res.step.params["metric"]
            df = res.result
            dcol = dim_column_of(df)
            rel = df[df["reliable"].fillna(False).astype(bool)] if "reliable" in df else df
            for _, r in rel.iterrows():
                if pd.isna(r.get(m)):
                    continue
                rows.append((abs(float(r["diff_vs_baseline"])),
                             ac.DIMENSIONS[res.step.params["dimension"]].label,
                             r[dcol], r[m], r["diff_vs_baseline"], r.get(f"{m}_n")))
        rows.sort(key=lambda x: -x[0])

        if rows:
            _, dlabel, seg, val, diff, n = rows[0]
            m = cmps[0].step.params["metric"]
            ans.headline = (f"The segment furthest from the book on "
                            f"{metric_label(m).lower()} is {dlabel} = **{seg}** at "
                            f"{fmt_metric(m, val)}, {gap_phrase(m, diff)} the "
                            f"portfolio baseline.")
            for _, dlabel, seg, val, diff, n in rows[:6]:
                ans.evidence.append(
                    f"**{dlabel} = {seg}**: {fmt_metric(m, val)} "
                    f"({gap_phrase(m, diff)} baseline, n={n:,.0f})")
        elif ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            r0 = ds.result.iloc[0]
            ans.headline = (f"The strongest association is {r0['dimension_label']} = "
                            f"**{r0['segment']}** at {fmt_metric(m, r0['value'])}.")
        else:
            ans.headline = "No segment cleared the sample-size threshold for this comparison."

        if ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            ans.analysis.append(
                f"A driver scan across {ds.result['dimension'].nunique()} dimensions "
                f"ranked every segment by deviation from the reference average. "
                f"Top signals: " + "; ".join(
                    f"{r['dimension_label']}={r['segment']} "
                    f"({fmt_metric(m, r['value'])}, n={r['n']})"
                    for _, r in ds.result.head(4).iterrows()) + ".")
        ans.analysis.append(
            "Each dimension was tested independently against the same baseline, so "
            "segments that overlap (for example income band and premium-to-income band) "
            "may be describing the same underlying policies.")
        ans.implication.append(
            "Segments with a large gap but a small `n` are not actionable on their own; "
            "the ones worth a campaign combine an elevated rate with enough policies "
            "for the fix to register in the portfolio number.")
        if ds and ds.is_frame and not ds.result.empty:
            adverse = adverse_driver(ds.result)
            if adverse is not None:
                m = ds.step.params["metric"]
                ans.recommendation.append(
                    f"The single highest-risk segment found is "
                    f"**{adverse['dimension_label']} = {adverse['segment']}** at "
                    f"{fmt_metric(m, adverse['value'])} (n={adverse['n']}).")
        ans.recommendation.append(
            "Build the target list from the intersection of the top two or three "
            "segments rather than from any one of them, and size it before committing "
            "retention capacity.")
        ans.caveats.append(
            "These are associations between a segment and its observed rate. The "
            "dataset records no reason codes, so none of them is established as a cause.")

    # ---- driver / diagnostic ------------------------------------------
    def _t_driver_analysis(self, plan, results, ans):
        ds = self._find(results, "driver_scan")
        ov = self._overview(results)
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]

        if ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            df = ds.result
            r0 = df.iloc[0]
            base = r0["focus_baseline"]
            ans.headline = (f"The strongest observed association with "
                            f"{metric_label(m).lower()} is {r0['dimension_label']} = "
                            f"**{r0['segment']}** at {fmt_metric(m, r0['value'])}, "
                            f"{gap_phrase(m, r0['diff_vs_baseline'])} the reference "
                            f"average of {fmt_metric(m, base)}.")
            for _, r in df.head(6).iterrows():
                ans.evidence.append(
                    f"**{r['dimension_label']} = {r['segment']}**: {fmt_metric(m, r['value'])} "
                    f"({gap_phrase(m, r['diff_vs_baseline'])} reference, n={r['n']}, "
                    f"{r['direction']})")
            ans.analysis.append(
                f"{df['dimension'].nunique()} dimensions were scanned; every segment "
                f"with at least {ds.step.params.get('min_n', 15)} records was scored by "
                f"its deviation from the reference average, and the largest deviations "
                f"are listed above.")
        elif cmps:
            return self._t_leaderboard(plan, results, ans)
        else:
            ans.headline = "No driver signal cleared the sample-size threshold."

        for res in cmps:
            m2 = res.step.params["metric"]
            d2 = res.result
            if d2.empty:
                continue
            c2 = dim_column_of(d2)
            r0 = d2.iloc[0]
            ans.analysis.append(
                f"By {ac.DIMENSIONS[res.step.params['dimension']].label.lower()}: "
                f"**{r0[c2]}** is highest at {fmt_metric(m2, r0[m2])}, "
                f"{gap_phrase(m2, r0['diff_vs_baseline'])} the baseline of "
                f"{fmt_metric(m2, r0['baseline'])}.")

        sp = self._find(results, "segment_profile")
        if sp and sp.is_frame and not sp.result.empty:
            d = sp.result.copy()
            d["absdiff"] = pd.to_numeric(d["diff"], errors="coerce").abs()
            for _, r in d.sort_values("absdiff", ascending=False).head(4).iterrows():
                ans.analysis.append(
                    f"{r['label']}: {r['focus_fmt']} in the focus group vs "
                    f"{r['portfolio_fmt']} portfolio-wide.")

        if ov:
            ans.implication.append(
                f"Against a portfolio lapse rate of "
                f"{fmt_metric('lapse_rate', ov.get('lapse_rate', {}).get('value'))} and "
                f"surrender rate of "
                f"{fmt_metric('surrender_rate', ov.get('surrender_rate', {}).get('value'))}, "
                f"only deviations of several percentage points on a reasonably sized "
                f"segment are worth an operational response.")
        if not ans.implication:
            ans.implication.append(
                "The scan tells you where the outcome concentrates, which is what a "
                "targeted intervention needs; it does not tell you what to change.")
        if ds and ds.is_frame and not ds.result.empty:
            adverse = adverse_driver(ds.result)
            if adverse is not None:
                m = ds.step.params["metric"]
                ans.recommendation.append(
                    f"Start with **{adverse['dimension_label']} = {adverse['segment']}** "
                    f"at {fmt_metric(m, adverse['value'])} - the largest adverse "
                    f"deviation found (n={adverse['n']}).")
        ans.recommendation.append(
            "Take the top two or three segments and test them against each other in a "
            "cross-tab before committing - a single-dimension signal is often a proxy "
            "for another dimension it correlates with.")
        ans.caveats.append(
            "Every figure above is an observed association. This dataset has no "
            "reason codes, no time-to-event data and no control group, so it cannot "
            "support a causal claim.")

    def _t_diagnostic(self, plan, results, ans):
        focus = plan.bindings or {}
        focus_label = ", ".join(f"{k}={v}" for k, v in focus.items()
                                if not isinstance(v, (list, dict))) or "the focus group"

        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        sp = self._find(results, "segment_profile")
        ds = self._find(results, "driver_scan")

        # 1. confirm the gap. Where several metrics were compared, lead with the
        #    one on which the focus group is genuinely *worse* - otherwise the
        #    headline reports the metric the group happens to be good at.
        candidates = []
        for res in cmps:
            m = res.step.params["metric"]
            df = res.result
            dcol = dim_column_of(df)
            hit = None
            for v in focus.values():
                if isinstance(v, str):
                    sub = df[df[dcol].astype(str) == v]
                    if not sub.empty:
                        hit = sub.iloc[0]
                        break
            if hit is None:
                continue
            rank = int(df.index[df[dcol].astype(str) == str(hit[dcol])][0]) + 1
            spec = ac.METRICS.get(m)
            diff = float(hit.get("diff_vs_baseline") or 0.0)
            base = abs(float(hit["baseline"])) or 1e-9
            # positive = adverse when high is bad, and vice versa
            adverse = diff if (spec and spec.higher_is_better is False) else -diff
            candidates.append((adverse / base, m, hit, rank, len(df), dcol))
            ans.evidence.append(
                f"**{hit[dcol]}** has a {metric_label(m).lower()} of "
                f"{fmt_metric(m, hit[m])} - rank {rank} of {len(df)} - "
                f"{gap_phrase(m, hit['diff_vs_baseline'])} the portfolio baseline "
                f"of {fmt_metric(m, hit['baseline'])} (n={hit.get(f'{m}_n', float('nan')):,.0f}).")

        candidates.sort(key=lambda t: -t[0])
        confirmed = candidates[0][1:] if candidates else None

        if confirmed:
            m, hit, rank, total, dcol = confirmed
            spec = ac.METRICS.get(m)
            diff = float(hit["diff_vs_baseline"])
            is_adverse = (diff > 0) if (spec and spec.higher_is_better is False) else (diff < 0)
            if is_adverse:
                ans.headline = (
                    f"**{hit[dcol]}** is materially worse than the book on "
                    f"{metric_label(m).lower()} ({fmt_metric(m, hit[m])} vs "
                    f"{fmt_metric(m, hit['baseline'])}, rank {rank} of {total}); "
                    f"the gap concentrates in the segments below.")
            else:
                ans.headline = (
                    f"**{hit[dcol]}** is not actually worse than the book on any "
                    f"metric tested - the closest is {metric_label(m).lower()} at "
                    f"{fmt_metric(m, hit[m])} against a baseline of "
                    f"{fmt_metric(m, hit['baseline'])}.")
        else:
            ans.headline = f"Diagnostic decomposition of {focus_label}."

        # 2. where it concentrates
        if ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            for _, r in ds.result.head(5).iterrows():
                ans.evidence.append(
                    f"Inside the group, **{r['dimension_label']} = {r['segment']}** shows "
                    f"{fmt_metric(m, r['value'])} vs the group's own average of "
                    f"{fmt_metric(m, r['focus_baseline'])} "
                    f"({gap_phrase(m, r['diff_vs_baseline'])}, n={r['n']}).")
            ans.analysis.append(
                f"Step order: confirm the gap against peers, then scan "
                f"{ds.result['dimension'].nunique()} dimensions inside the group for "
                f"where the metric concentrates, then profile the group structurally.")

        # 3. sub-tables
        for res in self._find_all(results, "multi_metric_table"):
            df = res.result
            if not res.is_frame or df.empty:
                continue
            lead = headline_metric(df)
            c2 = dim_column_of(df)
            if lead is None or c2 is None:
                continue
            d = reliable_rows(df, lead).dropna(subset=[lead])
            if d.empty:
                continue
            hi = d.iloc[0]
            ans.analysis.append(
                f"{res.step.title}: **{hi[c2]}** leads on {metric_label(lead).lower()} "
                f"at {fmt_metric(lead, hi[lead])} "
                f"(n={hi.get(f'{lead}_n', float('nan')):,.0f}).")

        # 4. structural contrast
        if sp and sp.is_frame and not sp.result.empty:
            d = sp.result.copy()
            d["absdiff"] = pd.to_numeric(d["diff"], errors="coerce").abs()
            picks = d[d["metric"] != "policy_count"].sort_values("absdiff", ascending=False).head(4)
            for _, r in picks.iterrows():
                ans.analysis.append(
                    f"Structurally, {r['label'].lower()} is {r['focus_fmt']} here vs "
                    f"{r['portfolio_fmt']} across the book.")

        if confirmed:
            m, hit, rank, total, dcol = confirmed
            n = hit.get(f"{m}_n")
            if pd.notna(n) and ac.METRICS[m].unit == "%":
                excess = (float(hit[m]) - float(hit["baseline"])) / 100.0 * float(n)
                ans.implication.append(
                    f"If **{hit[dcol]}** performed at the book average, roughly "
                    f"{abs(excess):,.0f} fewer policies in this group would have gone "
                    f"the wrong way - that is the size of the prize.")
        if not ans.implication:
            ans.implication.append(
                "The decomposition localises the problem, which is what makes an "
                "intervention affordable: a targeted fix costs a fraction of a "
                "blanket one.")

        if ds and ds.is_frame and not ds.result.empty:
            r0 = adverse_driver(ds.result)
            if r0 is not None:
                ans.recommendation.append(
                    f"Investigate **{r0['dimension_label']} = {r0['segment']}** first - "
                    f"it carries the largest adverse deviation inside the group "
                    f"(n={r0['n']}).")
            else:
                ans.recommendation.append(
                    "No sub-segment inside the group performs materially worse than the "
                    "group average, so the weakness is diffuse rather than localised.")
        ans.recommendation.append(
            "Before acting, check whether the top segment is confounded with another "
            "dimension in the supporting cross-tabs; a single-factor signal often "
            "disappears once the mix is held constant.")
        ans.caveats.append(
            "This decomposition shows where the outcome sits, not why. The dataset "
            "carries no reason codes, no pricing data and no control group, so no "
            "causal conclusion is supported. Sub-group counts inside a single "
            "product, channel or zone are small.")

    # ---- comparisons ---------------------------------------------------
    def _t_comparison(self, plan, results, ans):
        tables = [r for r in self._find_all(results, "multi_metric_table") if r.is_frame]
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        if not tables:
            return self._t_leaderboard(plan, results, ans)

        primary = tables[0]
        df = primary.result
        dcol = dim_column_of(df)
        metrics = [c for c in df.columns if c in ac.METRICS or c in ac.SHARE_METRICS]

        highlight = next((m for m in ("persistency_rate", "lapse_rate", "loss_ratio",
                                      "retention_success_rate", "risk_claim_rate")
                          if m in metrics), metrics[0] if metrics else None)
        if highlight:
            d = df.dropna(subset=[highlight])
            better_high = ac.METRICS.get(highlight)
            asc = not (better_high and better_high.higher_is_better)
            ranked = d.sort_values(highlight, ascending=asc)
            best, worst = ranked.iloc[0], ranked.iloc[-1]
            ans.headline = (f"Across {len(df)} {dcol.replace('_', ' ')} groups and "
                            f"{len(metrics)} metrics, **{best[dcol]}** and "
                            f"**{worst[dcol]}** sit at opposite ends on "
                            f"{metric_label(highlight).lower()} "
                            f"({fmt_metric(highlight, best[highlight])} vs "
                            f"{fmt_metric(highlight, worst[highlight])}).")

        for _, row in df.iterrows():
            bits = [f"{metric_label(m).lower()} {fmt_metric(m, row[m])}"
                    for m in metrics[:7] if pd.notna(row.get(m))]
            ans.evidence.append(f"**{row[dcol]}**: " + "; ".join(bits))

        ans.analysis.append(
            "The comparison is multi-metric by construction: "
            + ", ".join(metric_label(m).lower() for m in metrics)
            + " were each computed on their own declared denominator, so no single "
              "aggregate drives the ranking.")
        for res in tables[1:]:
            d2 = res.result
            if d2.empty:
                continue
            c2 = dim_column_of(d2)
            m2 = [c for c in d2.columns if c in ac.METRICS or c in ac.SHARE_METRICS]
            if not m2:
                continue
            ans.analysis.append(
                f"{res.step.title}: **{d2.iloc[0][c2]}** leads on "
                f"{metric_label(m2[0]).lower()} at {fmt_metric(m2[0], d2.iloc[0][m2[0]])}.")
        for res in cmps:
            m3 = res.step.params["metric"]
            d3 = res.result
            if d3.empty:
                continue
            c3 = dim_column_of(d3)
            spread = float(d3[m3].max()) - float(d3[m3].min())
            ans.analysis.append(
                f"On {metric_label(m3).lower()} the spread between best and worst is "
                f"{gap_phrase(m3, spread).replace(' above', '').replace(' below', '')}, "
                f"against a portfolio baseline of {fmt_metric(m3, d3.iloc[0]['baseline'])}.")

        ans.implication.append(
            "Groups that lead on volume are not the same as the groups that lead on "
            "persistency or economics, so a decision taken on any one column alone "
            "would point the wrong way.")
        ans.recommendation.append(
            "Decide which column actually matters for the decision at hand, then check "
            "the `_n` columns - several of these groups are small enough that a few "
            "policies move the rate.")

    def _t_segment_contrast(self, plan, results, ans):
        profiles = [r for r in self._find_all(results, "segment_profile") if r.is_frame]
        if not profiles:
            return self._t_comparison(plan, results, ans)

        p0 = profiles[0].result.copy()
        focus_desc = _filters_text(profiles[0].step.params.get("filters"))
        p0["absdiff"] = pd.to_numeric(p0["diff"], errors="coerce").abs()
        rates = p0[p0["metric"].isin(["lapse_rate", "surrender_rate", "claim_rate",
                                      "loss_ratio", "persistency_rate"])]
        lead = rates.sort_values("absdiff", ascending=False).head(1)
        if not lead.empty:
            r = lead.iloc[0]
            direction = "higher" if float(r["diff"]) > 0 else "lower"
            ans.headline = (f"{focus_desc} differs most from the book on "
                            f"{r['label'].lower()}: {r['focus_fmt']} vs "
                            f"{r['portfolio_fmt']} ({direction}).")
        else:
            ans.headline = f"Profile of {focus_desc} against the whole portfolio."

        for _, r in p0.sort_values("absdiff", ascending=False).head(8).iterrows():
            flag = "" if r["reliable"] else "  *(small sample)*"
            ans.evidence.append(
                f"**{r['label']}**: {r['focus_fmt']} vs {r['portfolio_fmt']} "
                f"portfolio-wide (n={r['focus_n']:,.0f}){flag}")

        if len(profiles) > 1:
            p1 = profiles[1].result
            other = _filters_text(profiles[1].step.params.get("filters"))
            merged = p0[["metric", "label", "focus_fmt"]].merge(
                p1[["metric", "focus_fmt"]], on="metric", suffixes=("_a", "_b"))
            for _, r in merged.head(8).iterrows():
                ans.analysis.append(
                    f"{r['label']}: {focus_desc} {r['focus_fmt_a']} vs {other} "
                    f"{r['focus_fmt_b']}.")

        for res in self._find_all(results, "multi_metric_table"):
            df = res.result
            if not res.is_frame or df.empty:
                continue
            c2 = dim_column_of(df)
            lead = headline_metric(df)
            if lead is None or c2 is None:
                continue
            d = reliable_rows(df, lead).dropna(subset=[lead])
            if d.empty:
                continue
            ans.analysis.append(
                f"{res.step.title}: **{d.iloc[0][c2]}** leads on "
                f"{metric_label(lead).lower()} at "
                f"{fmt_metric(lead, d.iloc[0][lead])} "
                f"(n={d.iloc[0].get(f'{lead}_n', float('nan')):,.0f}).")

        ans.analysis.append(
            "Each metric is computed on its own denominator for the focus group and "
            "for the whole portfolio, so the two columns are directly comparable.")
        ans.implication.append(
            "Where the focus group differs on customer characteristics as well as on "
            "outcomes, the outcome gap may be a composition effect rather than "
            "something specific to the group itself.")
        ans.recommendation.append(
            "Use the largest differences as the definition for a target list, and "
            "verify against the cross-tabs that the difference is not simply a "
            "product- or channel-mix artefact.")

    def _t_grid(self, plan, results, ans):
        grids = [r for r in self._find_all(results, "crosstab_metric") if r.is_frame]
        if not grids:
            return self._t_leaderboard(plan, results, ans)

        g0 = grids[0]
        df = g0.result
        m = g0.step.params["metric"]
        a, b = g0.step.params["dim_a"], g0.step.params["dim_b"]
        ca, cb = ac.DIMENSIONS[a].column, ac.DIMENSIONS[b].column
        worst, best = df.iloc[0], df.iloc[-1]

        ans.headline = (f"The weakest {ac.DIMENSIONS[a].label.lower()} x "
                        f"{ac.DIMENSIONS[b].label.lower()} cell on "
                        f"{metric_label(m).lower()} is **{worst[ca]} / {worst[cb]}** at "
                        f"{fmt_metric(m, worst[m])} (n={worst['n']:,.0f}).")

        for _, r in df.head(6).iterrows():
            ans.evidence.append(
                f"**{r[ca]} / {r[cb]}**: {fmt_metric(m, r[m])} (n={r['n']:,.0f})")
        ans.evidence.append(
            f"Best cell: **{best[ca]} / {best[cb]}** at {fmt_metric(m, best[m])} "
            f"(n={best['n']:,.0f}).")

        ans.analysis.append(
            f"{len(df)} cells cleared the minimum sample size of "
            f"{g0.step.params.get('min_n', 10)}; cells below it were dropped rather "
            f"than ranked, so a three-policy pocket cannot appear as the worst cell.")
        for res in grids[1:]:
            d2 = res.result
            if d2.empty:
                continue
            a2, b2 = res.step.params["dim_a"], res.step.params["dim_b"]
            c2a, c2b = ac.DIMENSIONS[a2].column, ac.DIMENSIONS[b2].column
            m2 = res.step.params["metric"]
            ans.analysis.append(
                f"{res.step.title}: worst cell is **{d2.iloc[0][c2a]} / "
                f"{d2.iloc[0][c2b]}** at {fmt_metric(m2, d2.iloc[0][m2])} "
                f"(n={d2.iloc[0]['n']:,.0f}).")

        ov = self._overview(results)
        if ov and m in ov:
            ans.implication.append(
                f"Against a portfolio {metric_label(m).lower()} of "
                f"{ov[m]['formatted']}, the worst cell runs at {fmt_metric(m, worst[m])} - "
                f"a pocket this specific is exactly what a targeted intervention can reach.")
        else:
            ans.implication.append(
                "Two-way cells localise a problem far better than either dimension "
                "alone, which is what makes an intervention affordable.")
        ans.recommendation.append(
            f"Treat the top cells as investigation leads: check whether **{worst[ca]} / "
            f"{worst[cb]}** is a genuine pocket or the product of one shared "
            f"characteristic - the sample sizes are small even after filtering.")
        ans.caveats.append(
            "Two-way cells divide an already modest dataset. Even the cells shown here "
            "carry tens rather than hundreds of policies.")

    # ---- economics -----------------------------------------------------
    def _t_economics(self, plan, results, ans):
        tables = [r for r in self._find_all(results, "multi_metric_table") if r.is_frame]
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        comp = self._find(results, "composite_score")

        primary = next((t for t in tables if "loss_ratio" in t.result.columns), None)
        if comp is not None and primary is None:
            return self._t_priority_ranking(plan, results, ans)
        if primary is None:
            return self._t_leaderboard(plan, results, ans)

        df = primary.result
        dcol = dim_column_of(df)
        d = df.dropna(subset=["loss_ratio"]).sort_values("loss_ratio", ascending=False)
        worst, best = d.iloc[0], d.iloc[-1]

        has_risk = "risk_loss_ratio" in df.columns
        ans.headline = (f"**{worst[dcol]}** has the weakest claims economics with an "
                        f"all-claims loss ratio of {fmt_metric('loss_ratio', worst['loss_ratio'])}"
                        + (f", but only {fmt_metric('risk_loss_ratio', worst['risk_loss_ratio'])} "
                           f"of that is underwriting loss." if has_risk and pd.notna(worst.get("risk_loss_ratio"))
                           else "."))

        for _, r in d.head(6).iterrows():
            bits = [f"loss ratio {fmt_metric('loss_ratio', r['loss_ratio'])}"]
            if has_risk and pd.notna(r.get("risk_loss_ratio")):
                bits.append(f"risk-only {fmt_metric('risk_loss_ratio', r['risk_loss_ratio'])}")
            for extra in ("total_premium_collected", "total_claim_amount"):
                if extra in r and pd.notna(r[extra]):
                    bits.append(f"{metric_label(extra).lower()} {fmt_inr(r[extra])}")
            if "claim_rate" in r and pd.notna(r["claim_rate"]):
                bits.append(f"claim rate {fmt_metric('claim_rate', r['claim_rate'])}")
            ans.evidence.append(f"**{r[dcol]}**: " + "; ".join(bits))
        ans.evidence.append(
            f"Strongest economics: **{best[dcol]}** at "
            f"{fmt_metric('loss_ratio', best['loss_ratio'])}.")

        ans.analysis.append(ac.METRICS["loss_ratio"].definition)
        if has_risk:
            ans.analysis.append(ac.METRICS["risk_loss_ratio"].definition)
        for res in cmps:
            m = res.step.params["metric"]
            d2 = res.result
            if d2.empty:
                continue
            c2 = dim_column_of(d2)
            ans.analysis.append(
                f"Against the portfolio, **{d2.iloc[0][c2]}** is "
                f"{gap_phrase(m, d2.iloc[0]['diff_vs_baseline'])} the baseline "
                f"{metric_label(m).lower()} of {fmt_metric(m, d2.iloc[0]['baseline'])}.")

        grid = self._find(results, "crosstab_metric")
        if grid and grid.is_frame and not grid.result.empty:
            a, b = grid.step.params["dim_a"], grid.step.params["dim_b"]
            ca, cb = ac.DIMENSIONS[a].column, ac.DIMENSIONS[b].column
            g = grid.result
            ans.analysis.append(
                "Claim mix matters: " + "; ".join(
                    f"{r[ca]} - {r[cb]} ({int(r[grid.step.params['metric']])})"
                    for _, r in g.head(5).iterrows()) + ".")

        if has_risk and pd.notna(worst.get("risk_loss_ratio")):
            gap = float(worst["loss_ratio"]) - float(worst["risk_loss_ratio"])
            ans.implication.append(
                f"For **{worst[dcol]}**, {gap:.2f}x of the {float(worst['loss_ratio']):.2f}x "
                f"outflow is contractual (maturity and surrender payouts), not "
                f"underwriting loss. Judging the product on the headline ratio alone "
                f"would misread it.")
        ans.implication.append(
            "Loss ratio here measures claims against premium collected to date. On a "
            "book where many policies have not yet matured, a low ratio partly means "
            "'not yet paid out', not 'cheap'.")
        ans.recommendation.append(
            f"Review pricing and underwriting on **{worst[dcol]}** using the risk-only "
            f"ratio, and confirm the claim-type mix before concluding anything about "
            f"product economics.")
        ans.caveats.append(
            "This dataset holds no expense, commission, reserve or investment-return "
            "data. Everything above is claims-to-premium economics, not profitability.")

    # ---- concentration / anomaly ---------------------------------------
    def _t_concentration(self, plan, results, ans):
        concs = [r for r in self._find_all(results, "concentration")
                 if isinstance(r.result, dict)]
        tops = [r for r in self._find_all(results, "top_entity_concentration")
                if isinstance(r.result, dict)]

        if not concs and not tops:
            return self._t_leaderboard(plan, results, ans)

        if concs:
            worst = max(concs, key=lambda r: r.result["hhi"])
            c = worst.result
            ans.headline = (f"The book is most concentrated by **{c['dimension']}**: "
                            f"HHI {c['hhi']} ({c['interpretation']}), equivalent to only "
                            f"{c['effective_n']} effective groups out of {c['n_groups']}.")
            for r in concs:
                cc = r.result
                ans.evidence.append(
                    f"**By {cc['dimension']}** ({metric_label(cc['value_metric']).lower()}): "
                    f"HHI {cc['hhi']} - {cc['interpretation']}; top {cc['top_n']} hold "
                    f"{cc['top_n_share_pct']:.1f}% across {cc['n_groups']} groups.")
        for r in tops:
            t = r.result
            ans.evidence.append(
                f"**Largest {t['top_n']} records by {t['column'].replace('_', ' ')}**: "
                f"{t['top_n_share_pct']:.1f}% of the total ({fmt_inr(t['top_n_value'])} "
                f"of {fmt_inr(t['total'])} across {t['n_records']:,} records).")
        if not ans.headline and tops:
            t = tops[0].result
            ans.headline = (f"The largest {t['top_n']} records carry "
                            f"{t['top_n_share_pct']:.1f}% of total "
                            f"{t['column'].replace('_', ' ')}.")

        ans.analysis.append(
            "Concentration is measured with the Herfindahl-Hirschman Index (sum of "
            "squared shares). Above 0.25 is treated as highly concentrated, 0.15-0.25 "
            "as moderately concentrated. `effective_n` is 1/HHI - the number of "
            "equal-sized groups the actual distribution is equivalent to.")
        for res in self._find_all(results, "multi_metric_table"):
            if not res.is_frame or res.result.empty:
                continue
            c2 = dim_column_of(res.result)
            m2 = [c for c in res.result.columns if c in ac.METRICS or c in ac.SHARE_METRICS]
            if m2 and c2:
                ans.analysis.append(
                    f"{res.step.title}: **{res.result.iloc[0][c2]}** leads on "
                    f"{metric_label(m2[0]).lower()}.")

        ans.implication.append(
            "Concentration is not a defect by itself - it becomes one when the "
            "concentrated group also underperforms, because there is then nothing "
            "else in the book to absorb it.")
        ans.recommendation.append(
            "Cross-reference the most concentrated dimension against the persistency "
            "and loss-ratio tables: concentration in a strong performer is a strategy, "
            "concentration in a weak one is an exposure.")

    def _t_anomaly(self, plan, results, ans):
        outs = [r for r in self._find_all(results, "outliers") if r.is_frame]
        concs = [r for r in self._find_all(results, "concentration")
                 if isinstance(r.result, dict)]
        tops = [r for r in self._find_all(results, "top_entity_concentration")
                if isinstance(r.result, dict)]
        shorts = [r for r in self._find_all(results, "entity_shortlist") if r.is_frame]

        flagged_total = 0
        lines: List[str] = []
        for res in outs:
            df = res.result
            m = res.step.params["metric"]
            dname = res.step.params["dimension"]
            dcol = ac.DIMENSIONS[dname].column
            flagged = df[df["is_outlier"].fillna(False).astype(bool)]
            flagged_total += len(flagged)
            if flagged.empty:
                lines.append(
                    f"**{ac.DIMENSIONS[dname].label} / {metric_label(m).lower()}**: no "
                    f"outliers - all {len(df)} groups fall inside "
                    f"{df['method'].iloc[0] if 'method' in df else 'the fence'}.")
            else:
                names = ", ".join(
                    f"{r[dcol]} ({fmt_metric(m, r[m])}, n={r.get(f'{m}_n', float('nan')):,.0f})"
                    for _, r in flagged.head(5).iterrows())
                lines.append(
                    f"**{ac.DIMENSIONS[dname].label} / {metric_label(m).lower()}**: "
                    f"{len(flagged)} outlier(s) - {names}.")

        if flagged_total:
            ans.headline = (f"{flagged_total} statistical outlier(s) were flagged across "
                            f"{len(outs)} tests; each is a lead to review, not a finding.")
        elif concs or tops:
            ans.headline = ("No statistical outliers were flagged; the notable pattern "
                            "is concentration rather than deviation.")
        else:
            ans.headline = "No unusual patterns cleared the detection thresholds."

        ans.evidence.extend(lines)
        for r in concs:
            c = r.result
            ans.evidence.append(
                f"**Concentration by {c['dimension']}**: HHI {c['hhi']} "
                f"({c['interpretation']}); top {c['top_n']} hold {c['top_n_share_pct']:.1f}%.")
        for r in tops:
            t = r.result
            ans.evidence.append(
                f"**Top {t['top_n']} records by {t['column'].replace('_', ' ')}**: "
                f"{t['top_n_share_pct']:.1f}% of the total.")
        for r in shorts:
            df = r.result
            if df.empty:
                continue
            idcol = df.columns[0]
            ans.evidence.append(
                f"**{r.step.title}**: top entries are "
                + ", ".join(str(v) for v in df[idcol].head(5).tolist()) + ".")

        ans.analysis.append(
            "Two detection methods are used. The IQR fence flags a group outside "
            "Q1-1.5*IQR to Q3+1.5*IQR; the z-score test flags anything beyond two "
            "standard deviations. Groups below the minimum sample size are excluded "
            "before either test runs.")
        ans.implication.append(
            "Outliers on small books are common and mostly benign - an agent with 10 "
            "policies moves 10 percentage points on one lapse. Concentration findings "
            "are structurally more serious because they persist.")
        ans.recommendation.append(
            "Review the flagged items operationally before drawing any conclusion, and "
            "prioritise concentration findings over individual outliers when deciding "
            "what to escalate.")
        ans.caveats.append(
            "These are statistical flags on a small dataset. Nothing here evidences "
            "misconduct, mis-selling or data error - each flag needs verification "
            "against the underlying records.")

    # ---- distribution ---------------------------------------------------
    def _t_distribution(self, plan, results, ans):
        ov = self._overview(results)
        tables = [r for r in self._find_all(results, "multi_metric_table") if r.is_frame]

        if ov:
            mix = ov.get("_status_mix") or {}
            top = max(mix.items(), key=lambda kv: kv[1]) if mix else None
            ans.headline = (f"The book is {top[1]:.1f}% {top[0]} across "
                            f"{ov['policy_count']['value']:,.0f} policies."
                            if top else "Portfolio distribution.")
            if mix:
                ans.evidence.append("**Policy status mix**: " + ", ".join(
                    f"{k} {v:.1f}% ({ov['_status_counts'][k]:,})"
                    for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))
        for res in tables:
            df = res.result
            c2 = dim_column_of(df)
            share = next((c for c in ("policy_share", "sum_assured_share", "premium_share")
                          if c in df.columns), None)
            if not share or c2 is None:
                continue
            d = df.sort_values(share, ascending=False)
            ans.evidence.append(
                f"**{res.step.title}**: " + ", ".join(
                    f"{r[c2]} {r[share]:.1f}%" for _, r in d.head(6).iterrows()))
            ans.analysis.append(
                f"{res.step.title} - largest group **{d.iloc[0][c2]}** at "
                f"{d.iloc[0][share]:.1f}% of {metric_label(share).lower().replace('share of portfolio ', '')}.")
        if not ans.headline:
            ans.headline = "Portfolio distribution across the requested dimensions."
        ans.analysis.append(
            "Volume share and value share are reported separately because they "
            "diverge: a product can be a small share of policies and a large share of "
            "sum assured, or the reverse.")
        ans.implication.append(
            "Mix is the denominator for every other judgement in this system - a "
            "problem in a 3%-of-book segment and the same problem in a 70%-of-book "
            "segment are not the same problem.")
        ans.recommendation.append(
            "Read this alongside the persistency and economics tables; the mix only "
            "becomes a decision once you know which parts of it perform.")

    # ---- shortlists ------------------------------------------------------
    def _t_shortlist(self, plan, results, ans):
        shorts = [r for r in self._find_all(results, "entity_shortlist") if r.is_frame]
        if not shorts:
            return self._t_leaderboard(plan, results, ans)

        s0 = shorts[0]
        df = s0.result
        base = s0.step.params.get("base", "policy")
        idcol = df.columns[0]
        namecol = next((c for c in ("customer_name", "full_name") if c in df.columns), None)

        ans.headline = (f"{len(df)} {base} records are ranked below by an explicit "
                        f"scoring framework; the top entry is **{df.iloc[0][idcol]}**"
                        + (f" ({df.iloc[0][namecol]})" if namecol else "") + ".")

        value_cols = [c for c in ("total_annual_premium", "annual_premium", "sum_assured",
                                  "total_sum_assured", "premium_to_income",
                                  "observed_lapse_rate", "attrition_ratio",
                                  "target_achievement_pct", "policies_serviced",
                                  "sa_to_income", "total_premium_paid")
                      if c in df.columns]
        for _, r in df.head(6).iterrows():
            bits = []
            if namecol:
                bits.append(str(r[namecol]))
            for c in value_cols[:4]:
                v = r[c]
                if pd.isna(v):
                    continue
                if c in ("premium_to_income", "sa_to_income"):
                    bits.append(f"{c.replace('_', '-')} {float(v):.2f}")
                elif "rate" in c or "ratio" in c or c.endswith("_pct"):
                    bits.append(f"{c.replace('_', ' ')} {float(v):.2f}")
                elif c in ("policies_serviced",):
                    bits.append(f"{int(v)} policies")
                else:
                    bits.append(f"{c.replace('_', ' ')} {fmt_inr(v)}")
            score = r.get("priority_score")
            tail = f" [score {score:.3f}]" if pd.notna(score) else ""
            ans.evidence.append(f"**{r[idcol]}**: " + "; ".join(bits) + tail)

        comps = s0.step.params.get("score_components") or []
        if comps:
            ans.analysis.append(
                "Ranking framework: " + ", ".join(
                    f"{c['column'].replace('_', ' ')} "
                    f"({int(float(c.get('weight', 1)) * 100)}%, "
                    f"{'higher' if c.get('direction', 'high') == 'high' else 'lower'} scores higher)"
                    for c in comps)
                + ". Each column was min-max normalised across the filtered population "
                  "and weighted; there is no model and no probability involved.")
        if s0.step.params.get("filters"):
            ans.analysis.append(
                f"Population filtered to: {_filters_text(s0.step.params['filters'])}.")

        for res in self._find_all(results, "compare_to_baseline"):
            m = res.step.params["metric"]
            d2 = res.result
            if not res.is_frame or d2.empty:
                continue
            c2 = dim_column_of(d2)
            ans.analysis.append(
                f"Context - **{d2.iloc[0][c2]}** carries the highest "
                f"{metric_label(m).lower()} at {fmt_metric(m, d2.iloc[0][m])}, "
                f"{gap_phrase(m, d2.iloc[0]['diff_vs_baseline'])} the book average.")
        ds = self._find(results, "driver_scan")
        if ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            ans.analysis.append(
                "Segments most associated with the outcome: " + "; ".join(
                    f"{r['dimension_label']}={r['segment']} ({fmt_metric(m, r['value'])})"
                    for _, r in ds.result.head(3).iterrows()) + ".")

        prem = next((c for c in ("annual_premium", "total_annual_premium") if c in df.columns), None)
        if prem:
            total = pd.to_numeric(df[prem], errors="coerce").sum()
            ans.implication.append(
                f"The {len(df)} records on this list carry {fmt_inr(total)} of "
                f"annualised premium between them - that is the value the "
                f"intervention is protecting.")
        else:
            ans.implication.append(
                "A ranked shortlist converts a portfolio-level finding into work a "
                "team can actually execute this week.")
        ans.recommendation.append(
            "Work the list from the top and record the outcome of each contact, so the "
            "next iteration can be scored on observed response rather than on "
            "value-at-stake alone.")
        ans.caveats.append(
            "This ranking uses characteristics observed in the data. It is not a "
            "trained propensity model and the score must not be read as a probability.")

    def _t_retention_shortlist(self, plan, results, ans):
        self._t_shortlist(plan, results, ans)
        for res in self._find_all(results, "multi_metric_table"):
            if not res.is_frame or "retention_success_rate" not in res.result.columns:
                continue
            d = res.result.dropna(subset=["retention_success_rate"])
            if d.empty:
                continue
            c2 = dim_column_of(d)
            best = d.sort_values("retention_success_rate", ascending=False).iloc[0]
            ans.recommendation.insert(0,
                f"Reach them through **{best[c2]}** - it currently saves "
                f"{fmt_metric('retention_success_rate', best['retention_success_rate'])} "
                f"of contacted cases, the highest of the outreach channels "
                f"(n={best.get('retention_success_rate_n', float('nan')):,.0f}).")
            break

    # ---- action plan ------------------------------------------------------
    def _t_action_plan(self, plan, results, ans):
        ov = self._overview(results)
        comp = self._find(results, "composite_score")
        ds = self._find(results, "driver_scan")
        tables = [r for r in self._find_all(results, "multi_metric_table") if r.is_frame]

        actions: List[str] = []
        if comp and comp.is_frame and not comp.result.empty:
            df = comp.result
            dcol = dim_column_of(df)
            top = df.iloc[0]
            ans.headline = (f"Start with **{top[dcol]}** - it ranks first on a "
                            f"materiality-weighted framework (score "
                            f"{top['priority_score']:.3f}).")
            for _, r in df.head(3).iterrows():
                bits = [f"{metric_label(c['metric']).lower()} {fmt_metric(c['metric'], r.get(c['metric']))}"
                        for c in comp.step.params["components"] if c["metric"] in r]
                ans.evidence.append(f"**{r[dcol]}** (rank {int(r['rank'])}): " + "; ".join(bits))
            actions.append(
                f"Scope a targeted programme on **{top[dcol]}**, sized against the "
                f"portfolio share shown in the supporting table.")

        if ds and ds.is_frame and not ds.result.empty:
            m = ds.step.params["metric"]
            for _, r in ds.result.head(4).iterrows():
                ans.evidence.append(
                    f"**{r['dimension_label']} = {r['segment']}**: {fmt_metric(m, r['value'])} "
                    f"({gap_phrase(m, r['diff_vs_baseline'])} the reference average, n={r['n']})")
            r0 = adverse_driver(ds.result)
            if r0 is not None:
                actions.append(
                    f"Investigate **{r0['dimension_label']} = {r0['segment']}** - the "
                    f"largest adverse deviation found, at {fmt_metric(m, r0['value'])} "
                    f"against {fmt_metric(m, r0['focus_baseline'])} (n={r0['n']}).")

        for res in tables:
            df = res.result
            if "retention_success_rate" not in df.columns:
                continue
            d = df.dropna(subset=["retention_success_rate"])
            if d.empty:
                continue
            c2 = dim_column_of(d)
            best = d.sort_values("retention_success_rate", ascending=False).iloc[0]
            worst = d.sort_values("retention_success_rate").iloc[0]
            ans.evidence.append(
                f"Retention today saves {fmt_metric('retention_success_rate', best['retention_success_rate'])} "
                f"via **{best[c2]}** and only "
                f"{fmt_metric('retention_success_rate', worst['retention_success_rate'])} "
                f"via **{worst[c2]}**.")
            actions.append(
                f"Shift outreach volume from **{worst[c2]}** towards **{best[c2]}**, "
                f"which has the higher observed save rate.")
            break

        out = self._find(results, "outliers")
        if out and out.is_frame and not out.result.empty:
            flagged = out.result[out.result["is_outlier"].fillna(False).astype(bool)]
            if not flagged.empty:
                dcol = ac.DIMENSIONS[out.step.params["dimension"]].column
                actions.append(
                    f"Review the {len(flagged)} statistical outlier(s) flagged on "
                    f"{metric_label(out.step.params['metric']).lower()}: "
                    + ", ".join(str(v) for v in flagged[dcol].head(4).tolist()) + ".")

        tec = self._find(results, "top_entity_concentration")
        if tec and isinstance(tec.result, dict):
            t = tec.result
            actions.append(
                f"Confirm the largest {t['top_n']} exposures individually - they carry "
                f"{t['top_n_share_pct']:.1f}% of total {t['column'].replace('_', ' ')}.")

        if not ans.headline:
            ans.headline = "Prioritised action list based on the evidence below."

        if ov:
            ans.analysis.append(
                f"Baselines: lapse {fmt_metric('lapse_rate', ov.get('lapse_rate', {}).get('value'))}, "
                f"surrender {fmt_metric('surrender_rate', ov.get('surrender_rate', {}).get('value'))}, "
                f"retention success "
                f"{fmt_metric('retention_success_rate', ov.get('retention_success_rate', {}).get('value'))}, "
                f"loss ratio {fmt_metric('loss_ratio', ov.get('loss_ratio', {}).get('value'))}.")
            lapse = ov.get("lapse_rate", {})
            if lapse.get("value") and lapse.get("denominator"):
                ans.implication.append(
                    f"Every percentage point of lapse on the exposed book of "
                    f"{lapse['denominator']:,.0f} policies is about "
                    f"{lapse['denominator'] / 100:,.0f} policies, so a one-point "
                    f"improvement is a measurable, not a cosmetic, outcome.")
        ans.analysis.append(
            "Actions are ordered by the size of the gap to baseline weighted by how "
            "much of the portfolio the group represents, not by the raw rate.")
        ans.recommendation.extend(actions or [
            "No group deviated far enough from the baseline to justify a dedicated "
            "programme; continue monitoring the KPI block."])
        ans.caveats.append(
            "The evidence identifies where outcomes concentrate. It does not identify "
            "why, so each action above is framed as investigate-and-target rather than "
            "as a fix.")



    def _t_prescriptive(self, plan, results, ans):
        """Template for PRESCRIPTIVE intent: 'How to improve X'.

        Ranks actionable levers by ROI = gap_size × group_volume, frames each
        as a concrete investigation target, and explains the evidence behind it.
        """
        ds = self._find(results, "driver_scan")
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        ov = self._overview(results)

        # Collect all segment-level gaps across all compare_to_baseline calls
        levers: List[Tuple[float, str, str, str, float, float, int]] = []
        # (roi_score, metric, dim_label, seg_label, value, diff, n)
        for res in cmps:
            m = res.step.params["metric"]
            df = res.result
            dcol = dim_column_of(df)
            if dcol is None:
                continue
            spec = ac.METRICS.get(m)
            rel = df[df["reliable"].fillna(False).astype(bool)] if "reliable" in df else df
            for _, r in rel.iterrows():
                diff = float(r.get("diff_vs_baseline", 0) or 0)
                n = int(r.get("n", r.get("sample_size", 0)) or 0)
                # adverse = going the wrong direction
                is_adverse = (diff < 0) if (spec and spec.higher_is_better) else (diff > 0)
                if not is_adverse or n == 0:
                    continue
                roi = abs(diff) * n  # gap × volume = impact proxy
                dim_label = ac.DIMENSIONS[res.step.params["dimension"]].label \
                    if res.step.params.get("dimension") in ac.DIMENSIONS else \
                    res.step.params.get("dimension", "").replace("_", " ").title()
                levers.append((roi, m, dim_label, str(r[dcol]), float(r[m] if m in r else r["value"]),
                               diff, n))

        # Also pull from driver_scan if available
        if ds and ds.is_frame and not ds.result.empty:
            m_ds = ds.step.params["metric"]
            for _, r in ds.result[ds.result["direction"] == "worse"].iterrows():
                roi = abs(float(r.get("diff_vs_baseline", 0))) * int(r.get("n", 1))
                levers.append((roi, m_ds, r["dimension_label"], str(r["segment"]),
                               float(r["value"]), float(r["diff_vs_baseline"]), int(r["n"])))

        levers.sort(key=lambda x: -x[0])

        # Deduplicate by (metric, segment)
        seen, unique_levers = set(), []
        for lev in levers:
            key = (lev[1], lev[3])
            if key not in seen:
                seen.add(key)
                unique_levers.append(lev)

        top = unique_levers[:8]

        # Check for explicit dimension query (e.g., area/state or product ranking)
        q_lower = (plan.question or "").lower()
        wants_lowest = bool(re.search(r"\b(lowest|worst|bottom|least|poor|poorest|underperforming|lagging|minimum|min)\b", q_lower))
        dim_lead_str = None

        if cmps:
            first_cmp = cmps[0]
            c_df = first_cmp.result
            c_m = first_cmp.step.params.get("metric", "persistency_rate")
            c_dcol = dim_column_of(c_df)
            if c_dcol and not c_df.empty:
                c_rel = c_df[c_df["reliable"].fillna(False).astype(bool)] if "reliable" in c_df else c_df
                c_rel = c_rel if not c_rel.empty else c_df
                c_sorted_df = c_df.sort_values(by=c_m, ascending=True)
                c_sorted_rel = c_rel.sort_values(by=c_m, ascending=True)
                c_obs_lowest = c_sorted_df.iloc[0]
                c_rel_lowest = c_sorted_rel.iloc[0]
                obs_n = c_obs_lowest.get(f'{c_m}_n', c_obs_lowest.get('sample_size', 1))
                rel_n = c_rel_lowest.get(f'{c_m}_n', c_rel_lowest.get('sample_size', 1))
                dim_title = ac.DIMENSIONS[first_cmp.step.params.get("dimension")].label if first_cmp.step.params.get("dimension") in ac.DIMENSIONS else c_dcol.replace("_", " ").title()

                if wants_lowest or any(w in q_lower for w in ("which area", "which state", "which product", "which channel")):
                    if str(c_obs_lowest[c_dcol]) != str(c_rel_lowest[c_dcol]) and pd.notna(obs_n) and obs_n < 20:
                        dim_lead_str = (f"**{c_obs_lowest[c_dcol]}** has the lowest observed {metric_label(c_m).lower()} at "
                                        f"{fmt_metric(c_m, c_obs_lowest[c_m])} (n={obs_n:,.0f}, below reliability threshold); "
                                        f"among reliable {dim_title.lower()}s (n >= 30), **{c_rel_lowest[c_dcol]}** is lowest at {fmt_metric(c_m, c_rel_lowest[c_m])} (n={rel_n:,.0f}).")
                    else:
                        dim_lead_str = (f"**{c_rel_lowest[c_dcol]}** has the lowest {metric_label(c_m).lower()} at "
                                        f"{fmt_metric(c_m, c_rel_lowest[c_m])} ({gap_phrase(c_m, c_rel_lowest['diff_vs_baseline'])} baseline, n={rel_n:,.0f}).")

        if top:
            roi, m, dim_label, seg, val, diff, n = top[0]
            if dim_lead_str:
                ans.headline = f"{dim_lead_str} {len(top)} intervention lever(s) identified to improve {metric_label(m).lower()}."
                ans.evidence.append(f"📍 **Direct Finding**: {dim_lead_str}")
            else:
                ans.headline = (
                    f"{len(top)} intervention lever(s) identified to improve "
                    f"{metric_label(m).lower()} — highest ROI: "
                    f"**{dim_label} = {seg}** ({fmt_metric(m, val)}, n={n:,})."
                )
        else:
            ans.headline = dim_lead_str or "No segment cleared the gap threshold — the metric is uniformly distributed."
            if dim_lead_str:
                ans.evidence.append(f"📍 **Direct Finding**: {dim_lead_str}")

        for i, (roi, m, dim_label, seg, val, diff, n) in enumerate(top, 1):
            ans.evidence.append(
                f"**Lever {i}: {dim_label} = {seg}** — "
                f"{metric_label(m).lower()} is {fmt_metric(m, val)} "
                f"({gap_phrase(m, diff)} portfolio average, n={n:,}, "
                f"impact proxy = {roi:,.0f} policy-points)."
            )

        if ov:
            for m_key in ("lapse_rate", "persistency_rate", "persistency_13m_rate",
                          "surrender_rate", "pivc_mismatch_rate"):
                v = ov.get(m_key, {}).get("value")
                n = ov.get(m_key, {}).get("denominator", 0)
                if v is not None and pd.notna(v):
                    ans.analysis.append(
                        f"Portfolio baseline {metric_label(m_key).lower()}: "
                        f"{fmt_metric(m_key, v)} across {n:,.0f} exposed policies."
                    )
                    break

        ans.analysis.append(
            "Levers are ranked by **impact proxy** = |gap to portfolio average| × "
            "group size. A large gap on a tiny group is not actionable; a small gap "
            "on a large group moves the aggregate number. Use this ranking to sequence "
            "interventions, not as a measure of severity alone."
        )
        ans.analysis.append(
            "Each lever represents an association — not a proven cause. The data shows "
            "where the outcome is concentrated, but not why it is concentrated there."
        )

        # Build specific "how to" recommendations per top lever
        how_to_map = {
            "payment_mode":    "Migrate Cheque/Cash payers to ECS/auto-debit — payment friction is the most operationally tractable lever.",
            "plan_name":       "Review product-level onboarding, premium affordability, and renewal communication for this plan specifically.",
            "trad_ulip":       "Segment ULIPs vs traditional products in renewal campaigns — the dynamics are different.",
            "state":           "Deploy a geo-targeted renewal task force or regional bancassurance push in this state.",
            "occupation":      "Tailor renewal reminder frequency and channel (SMS vs call) to this occupation segment.",
            "income_band":     "Review premium-to-income ratio for this income band — affordability stress may be driving the lapse.",
            "age_band":        "Adjust communication tone and channel for this age group; younger cohorts respond better to digital.",
            "persistency_bucket": "Prioritise early-stage (13m) intervention — lapse risk peaks at the first renewal hurdle.",
            "sales_type":      "Investigate whether this agent type has a mis-selling pattern or inadequate post-sale servicing.",
            "channel":         "Re-evaluate the channel's onboarding quality — high lapse often signals a new-business quality issue.",
        }
        dims_used = set(lev[2].lower().replace(" ", "_") for lev in top[:5])
        for d in dims_used:
            if d in how_to_map:
                ans.recommendation.append(how_to_map[d])

        if not ans.recommendation:
            ans.recommendation.append(
                "Target the highest-impact lever first: scope a 90-day renewal "
                "campaign on the segment combination above, track save-rate weekly, "
                "and re-run this analysis at the next quarter boundary."
            )
        ans.recommendation.append(
            "Cross-reference the top two levers — if the same policies appear in both, "
            "that is the intersection to prioritise for retention effort."
        )
        ans.caveats.append(
            "Impact proxy = gap × volume is a proxy, not a revenue forecast. The dataset "
            "carries no reason codes, no contact history, and no experiment data, so none "
            "of these levers is proven to cause the outcome. Each needs operational "
            "investigation before committing budget."
        )

    def _t_correlation(self, plan, results, ans):
        """Template for CROSS_DOMAIN intent: 'Relation between X and Y'.

        Surfaces Pearson r, the direction of association, and the segments
        where both metrics are simultaneously adverse (combined-risk hotspots).
        """
        corr_results = [r for r in self._find_all(results, "correlation_analysis") if r.is_frame]
        cmps = [r for r in self._find_all(results, "compare_to_baseline") if r.is_frame]
        ov = self._overview(results)

        if not corr_results:
            # Fallback: if only compare_to_baseline steps ran, render as a
            # side-by-side comparison and note the limitation
            ans.headline = "Cross-domain comparison completed; correlation step was not executed."
            for res in cmps[:2]:
                m = res.step.params["metric"]
                df = res.result
                dcol = dim_column_of(df)
                if dcol and not df.empty:
                    r0 = df.iloc[0]
                    ans.evidence.append(
                        f"{metric_label(m)}: **{r0[dcol]}** leads at "
                        f"{fmt_metric(m, r0.get(m, r0['value']))} "
                        f"({gap_phrase(m, r0.get('diff_vs_baseline', 0))} baseline)."
                    )
            ans.analysis.append(
                "A full correlation analysis requires both metrics to share a common "
                "dimension. Check that the analytical plan includes a 'correlation_analysis' step."
            )
            ans.caveats.append(
                "Segment-level correlations are not the same as individual-policy correlations. "
                "The relationship observed here is ecological (group-level) rather than individual."
            )
            return

        res0 = corr_results[0]
        df = res0.result
        m_a = res0.step.params["metric_a"]
        m_b = res0.step.params["metric_b"]
        dim = res0.step.params.get("dimension", "plan_name")
        dcol = ac.DIMENSIONS[dim].column if dim in ac.DIMENSIONS else dim
        dim_label = ac.DIMENSIONS[dim].label if dim in ac.DIMENSIONS else dim.replace("_", " ").title()

        pearson_r = float(df["pearson_r"].iloc[0]) if "pearson_r" in df.columns else float("nan")
        direction = df["direction"].iloc[0] if "direction" in df.columns else "unknown"
        n_segs = int(df["n_segments"].iloc[0]) if "n_segments" in df.columns else len(df)

        # Headline
        if pd.isna(pearson_r):
            ans.headline = (
                f"Across {n_segs} {dim_label} groups, there is insufficient data "
                f"to compute a reliable correlation between {metric_label(m_a).lower()} "
                f"and {metric_label(m_b).lower()}."
            )
        else:
            strength = ("strong" if abs(pearson_r) >= 0.6 else
                        "moderate" if abs(pearson_r) >= 0.3 else "weak")
            ans.headline = (
                f"Across {n_segs} {dim_label} groups, {metric_label(m_a).lower()} and "
                f"{metric_label(m_b).lower()} show a **{strength} {direction}** association "
                f"(Pearson r = {pearson_r:.2f})."
            )

        # Evidence: both-adverse hotspots first
        both_adv = df[df["both_adverse"].fillna(False).astype(bool)] if "both_adverse" in df.columns else pd.DataFrame()
        if not both_adv.empty:
            for _, r in both_adv.head(5).iterrows():
                ans.evidence.append(
                    f"⚠️ **{r[dcol]}**: {metric_label(m_a).lower()} = {fmt_metric(m_a, r[m_a])}, "
                    f"{metric_label(m_b).lower()} = {fmt_metric(m_b, r[m_b])} "
                    f"— adverse on **both** metrics (n={r.get(f'{m_a}_n', 'N/A'):,.0f})."
                )
        else:
            for _, r in df.head(5).iterrows():
                ans.evidence.append(
                    f"**{r[dcol]}**: {metric_label(m_a).lower()} = {fmt_metric(m_a, r[m_a])}, "
                    f"{metric_label(m_b).lower()} = {fmt_metric(m_b, r[m_b])} "
                    f"(n={r.get(f'{m_a}_n', 'N/A'):,.0f})."
                )

        # Exceptions: segments where the pattern breaks
        if not pd.isna(pearson_r):
            spec_a = ac.METRICS.get(m_a)
            avg_a = df[m_a].mean()
            avg_b = df[m_b].mean()
            breaks = []
            for _, r in df.iterrows():
                goes_up_a = r[m_a] > avg_a
                goes_up_b = r[m_b] > avg_b
                if goes_up_a != goes_up_b:
                    breaks.append(str(r[dcol]))
            if breaks:
                ans.analysis.append(
                    f"The pattern breaks for {len(breaks)} group(s) — "
                    f"{', '.join(breaks[:4])} — where the two metrics move in opposite "
                    f"directions. These are worth investigating separately."
                )

        ans.analysis.append(
            f"Correlation is computed at the **{dim_label} level** across {n_segs} groups. "
            f"This is an ecological (group-level) correlation. It does not guarantee "
            f"that the same relationship holds for individual policies within any group."
        )
        if not pd.isna(pearson_r):
            ans.analysis.append(
                f"Pearson r = {pearson_r:.2f}: "
                + ("A value above +0.6 indicates the two metrics tend to rise together; "
                   if pearson_r > 0 else
                   "A negative value indicates one tends to be high when the other is low; ")
                + "but the association may be driven by a confounding dimension (e.g. product mix or channel)."
            )

        n_both = len(both_adv)
        if n_both > 0:
            ans.implication.append(
                f"The {n_both} group(s) flagged as adverse on both metrics are the highest-priority "
                f"combined-risk segments — poor performance on one amplifies the other."
            )
        if not pd.isna(pearson_r) and abs(pearson_r) >= 0.4:
            ans.implication.append(
                f"A {'positive' if pearson_r > 0 else 'negative'} r = {pearson_r:.2f} "
                f"suggests that efforts to improve {metric_label(m_a).lower()} may also "
                f"{'improve' if pearson_r > 0 else 'worsen'} {metric_label(m_b).lower()} — "
                f"but this is observational. Verify with an operational team before acting."
            )

        if n_both > 0:
            both_segs = both_adv[dcol].head(3).tolist()
            ans.recommendation.append(
                f"Prioritise the combined-risk groups: **{', '.join(str(s) for s in both_segs)}**. "
                f"These show adverse signals on both {metric_label(m_a).lower()} and "
                f"{metric_label(m_b).lower()} simultaneously."
            )
        ans.recommendation.append(
            "Run a driver scan within each combined-risk group to find the sub-segment "
            "where the adverse combination is most concentrated before designing an intervention."
        )
        ans.caveats.append(
            "Correlation ≠ causation. This analysis identifies co-movement at the "
            f"{dim_label.lower()} level and does not establish that one metric "
            "influences the other. The dataset contains no experimental controls."
        )
        ans.caveats.append(
            "Claims data has only 134 records, making claim-related correlations "
            "statistically fragile. Treat claim-based correlations as directional "
            "signals, not firm findings."
        )


def _filters_text(filters: Any) -> str:
    if not filters or not isinstance(filters, dict):
        return "the whole portfolio"
    bits = []
    for k, v in filters.items():
        label = ac.DIMENSIONS[k].label if k in ac.DIMENSIONS else k.replace("_", " ")
        if isinstance(v, dict):
            rng = ", ".join(f"{kk} {vv}" for kk, vv in v.items())
            bits.append(f"{label} ({rng})")
        elif isinstance(v, (list, tuple)):
            bits.append(f"{label} in {list(v)}")
        else:
            bits.append(f"{label} = {v}")
    return "; ".join(bits)


# ---------------------------------------------------------------------
# optional LLM polish (never a source of numbers)
# ---------------------------------------------------------------------

LLM_SYSTEM = (
    "You are ExcelSense, an expert AI analytics assistant specialized strictly in Life Insurance portfolio and policy data.\n"
    "Your goal is to answer the user's specific business question clearly, directly, and faithfully, using ONLY the provided verified analytics data.\n\n"
    "DOMAIN & GROUNDING GUARDRAILS:\n"
    "1. DOMAIN BOUNDARY: You ONLY answer questions related to Life Insurance (policies, claims, persistency, customer demographics, and agent/sales performance). If a question is outside the life insurance domain (e.g. general trivia, coding, health/auto insurance, entertainment, politics), you MUST politely refuse by stating you are strictly specialized in Life Insurance data analytics.\n"
    "2. NO NUMBER ALTERATION: You MUST NOT invent, estimate, alter, change, or re-round any numbers, percentages, counts, or currency values. Use the EXACT figures provided in the verified data.\n"
    "3. STRICT EVIDENCE BOUNDARY: You CANNOT introduce metrics, external facts, outside entities, competitor data, or recommendations that are not present in the verified evidence.\n"
    "4. GROUNDED RECOMMENDATIONS: Every recommendation you state must directly connect to the identified segment and supporting evidence provided.\n"
    "5. NO CAUSAL CLAIMS: Do not claim causality from statistical correlation or co-movement (use 'is associated with' or 'co-moves with', NEVER 'causes' or 'caused by').\n"
    "6. AMBIGUITY CLARITY: If a requested metric or term is ambiguous, explicitly state the exact metric interpretation used.\n"
    "7. CLEAN CONVERSATIONAL PROSE: Write 2-3 clear, fluent paragraphs. Do NOT output rigid headers like 'Answer:', 'Evidence:', or 'Recommendation:'. Weave findings naturally.\n"
    "8. NO SURROUNDING QUOTES: Do NOT wrap the entire answer or paragraphs in quotation marks.\n"
    "9. NO MARKDOWN BOLDING: Do NOT use markdown bolding with asterisks (do NOT use **text** or *text*). Output plain, clean text without any asterisks.\n"
)


def _clean_llm_text(text: Optional[str]) -> Optional[str]:
    """Strip wrapping quotes and all markdown bolding/italics asterisks from LLM output."""
    if not text:
        return text
    import re
    t = text.strip()
    while (t.startswith('"""') and t.endswith('"""')) or (t.startswith("'''") and t.endswith("'''")):
        t = t[3:-3].strip()
    while (t.startswith('"') and t.endswith('"')) or (t.startswith('“') and t.endswith('”')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    # Remove markdown bold/italic asterisks so no literal ** shows in final output
    t = re.sub(r"\*\*([^*]+?)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+?)\*", r"\1", t)
    t = t.replace("**", "").replace("<b>", "").replace("</b>", "")
    return t.strip()


def polish(answer: BusinessAnswer, api_key: Optional[str] = None,
           timeout: int = 120) -> Optional[str]:
    """Conversational narrative synthesis. Uses local Ollama model
    or Groq/xAI cloud API if configured. Returns None if unreachable."""
    import json as _json
    import urllib.request
    import ollama_client as _oc

    prompt = (
        f"USER QUESTION: {answer.question}\n\n"
        f"VERIFIED DATA & FINDINGS:\n"
        f"- Headline Result: {answer.headline}\n"
        f"- Evidence Points:\n" + "\n".join(f"  * {e}" for e in answer.evidence) + "\n"
        f"- Analysis Points:\n" + "\n".join(f"  * {a}" for a in answer.analysis) + "\n"
        f"- Key Recommendations:\n" + "\n".join(f"  * {r}" for r in answer.recommendation) + "\n"
        f"- Caveats / Notes:\n" + "\n".join(f"  * {c}" for c in answer.caveats) + "\n\n"
        "Write a natural, strictly grounded, complete answer directly answering the question without wrapping in quotes and without any markdown bolding asterisks (**)."
    )

    # 1. Try local Ollama first
    if _oc.is_ollama_available():
        res = _oc.generate_llm_response(prompt=prompt, system_prompt=LLM_SYSTEM, timeout=timeout)
        if res:
            return _clean_llm_text(res)

    # 2. Fall back to cloud API keys if present
    key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        return None
    if key.startswith("gsk_"):
        url, model = "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"
    else:
        url, model = "https://api.x.ai/v1/chat/completions", "grok-2-1212"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        req = urllib.request.Request(
            url, data=_json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = _json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            # `_strip_wrapping_quotes` was never defined; the local-Ollama branch
            # above uses `_clean_llm_text`, so the cloud branch would have raised
            # NameError the first time it ran.
            return _clean_llm_text(content)
    except Exception:
        return None


__all__ = ["BusinessAnswer", "AnswerComposer", "polish", "fmt_metric", "fmt_inr",
           "metric_label", "gap_phrase", "reliable_rows", "adverse_driver",
           "dim_column_of", "headline_metric"]
