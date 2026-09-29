using System.Diagnostics;
using Mahjong.Engine;

namespace Mahjong.Policy.Efficiency;

/// <summary>
/// Experimental two-draw lookahead. Search only re-ranks baseline candidates at the
/// same best shanten and never publishes a partially searched result.
/// </summary>
public static class EnhancedSearch
{
    private const int MaxRootCandidates = 4;
    private const double CompleteHandValue = 100.0;
    private const double ShantenValue = 100.0;
    private const double UkeireValue = 1.0;

    public static PolicyEvaluation Analyze(StateSnapshot state, PolicyEvaluation baseline,
        TimeSpan budget, CancellationToken cancellationToken = default, IRuleSet? rules = null, RankCalibration? calibration = null)
    {
        var original = baseline;
        long started = Stopwatch.GetTimestamp();
        bool Expired() => cancellationToken.IsCancellationRequested || Stopwatch.GetElapsedTime(started) >= budget;
        if (Expired()) return original;
        rules ??= new RiichiRuleSet();

        if (state.PublicStateConsistent && state.SeatInfoKnown &&
            (state.Legal.Can(ActionFlags.Riichi) ||
             state.Legal.PonCandidates.Count + state.Legal.ChiCandidates.Count + state.Legal.KanCandidates.Count > 0))
        {
            var opponent = new Mahjong.Policy.Opponents.OpponentModel();
            var experimental = new EfficiencyPolicy(
                opponent,
                new HeuristicDiscardPolicy(new DefaultWeightProvider(), opponent, new Mahjong.Policy.Placement.PlacementAdjuster(), rules),
                new EnhancedCallPolicy(rules),
                new EnhancedRiichiPolicy(rules),
                new HeuristicPushFoldPolicy(),
                rules);
            baseline = experimental.Analyze(state) with { Source = "enhanced" };
        }

        if (Expired()) return original;
        if (!state.PublicStateConsistent || state.OurRiichi || baseline.Choice.Kind != ActionKind.Discard || baseline.Candidates.Count < 2)
            return baseline;

        // A folding decision is retained, including its explanation.
        if (baseline.Choice.DiscardTile != baseline.Candidates[0].Discard)
            return baseline;

        if (calibration is not null)
        {
            var calibrated = calibration.Select(state, baseline);
            if (calibrated.Source == "enhanced-calibrated")
                return Expired() ? original : calibrated;
        }

        var wall = DiscardScorer.BuildVisibleWall(state);
        int bestShanten = baseline.Candidates[0].ShantenAfter;
        var rootCandidates = baseline.Candidates
            .Where(c => c.ShantenAfter == bestShanten)
            .Take(MaxRootCandidates)
            .ToArray();

        if (rootCandidates.Length < 2)
            return baseline;

        var ranked = new List<(ScoredDiscard Candidate, double FutureValue, double Utility)>(rootCandidates.Length);
        foreach (var candidate in rootCandidates)
        {
            if (Expired()) return original;

            var concealed = state.Hand.ToList();
            if (!concealed.Remove(candidate.Discard))
                continue;

            double futureValue = EvaluateNextDraws(state, concealed, wall, Expired);
            if (Expired()) return original;

            // Future shape is deliberately combined with the baseline score rather than
            // replacing it. Baseline already prices dora/yakuhai/yaku potential, visible
            // tile scarcity, placement and candidate-specific deal-in cost.
            double utility = candidate.Score + futureValue;
            ranked.Add((candidate, futureValue, utility));
        }

        if (Expired() || ranked.Count < 2)
            return original;

        var selected = ranked
            .OrderByDescending(x => x.Utility)
            .ThenBy(x => x.Candidate.DealInCost)
            .ThenByDescending(x => x.Candidate.Score)
            .ThenBy(x => x.Candidate.Discard.Id)
            .First();

        var reason = new Reason(
            "two-draw-search",
            $"two-draw utility={selected.Utility:F2} (baseline={selected.Candidate.Score:F2}, future={selected.FutureValue:F2}); {ranked.Count} complete candidates");

        return baseline with
        {
            ElapsedMilliseconds = Stopwatch.GetElapsedTime(started).TotalMilliseconds,
            Source = "enhanced",
            Choice = ActionChoice.Discard(
                selected.Candidate.Discard,
                reason.Display,
                (baseline.Choice.Steps ?? []).Append(reason).ToArray()) with
            {
                DiscardIsRed = selected.Candidate.IsRed
            }
        };
    }

    private static double EvaluateNextDraws(
        StateSnapshot state,
        List<Tile> concealed,
        Wall wall,
        Func<bool> expired)
    {
        double total = 0;
        int totalLive = 0;

        // Reuse one count buffer for the whole root candidate. This avoids constructing
        // a second Wall and copying its 34-entry Seen array for every possible draw.
        var nextSeen = wall.Seen.ToArray();

        for (int id = 0; id < Tile.Count34; id++)
        {
            if (expired()) return double.NegativeInfinity;

            int live = wall.LiveOf(id);
            if (live == 0)
                continue;

            var draw = Tile.FromId(id);
            concealed.Add(draw);
            try
            {
                var hand = Hand.FromTiles(concealed, state.OurMelds);
                int shanten = Shanten(hand, state.OurMelds.Count);
                if (shanten == -1)
                {
                    total += live * CompleteHandValue;
                }
                else
                {
                    nextSeen[id]++;
                    var nextWall = new Wall();
                    nextWall.ObserveCounts(nextSeen);

                    // Preserve the established lexicographic rule: never trade worse
                    // shanten merely for more ukeire at the second discard.
                    var best = UkeireEnumerator.Enumerate(hand, nextWall)
                        .OrderBy(c => c.ShantenAfter)
                        .ThenByDescending(c => c.WeightedCount)
                        .ThenBy(c => c.Discard.Id)
                        .First();

                    nextSeen[id]--;
                    total += live * (-ShantenValue * best.ShantenAfter + UkeireValue * best.WeightedCount);
                }
            }
            finally
            {
                concealed.RemoveAt(concealed.Count - 1);
                // If an exception occurs after incrementing, restore the shared buffer.
                if (nextSeen[id] > wall.Seen[id])
                    nextSeen[id] = wall.Seen[id];
            }

            totalLive += live;
        }

        return totalLive == 0 ? 0.0 : total / totalLive;
    }

    private static int Shanten(Hand hand, int meldCount)
    {
        var counts = hand.CloneCounts();
        int standard = ShantenCalculator.Standard(counts, meldCount);
        if (meldCount != 0)
            return standard;

        return Math.Min(
            standard,
            Math.Min(ShantenCalculator.Chiitoitsu(counts), ShantenCalculator.Kokushi(counts)));
    }
}
