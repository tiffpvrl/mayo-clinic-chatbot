# ─────────────────────────────────────────────────────────────────────────────
# fig1_kpi_summary.R  ·  Overall quality scorecard with 95% bootstrap CIs
#
# Output: fig1_kpi_summary.pdf  (3.5 × 2.8 in, ready to embed in poster)
# ─────────────────────────────────────────────────────────────────────────────

source("data.R")

# ── Compute per-metric CI ─────────────────────────────────────────────────────
kpi <- quality_long |>
  group_by(metric_label) |>
  group_modify(~ boot_ci(.x$score)) |>
  ungroup() |>
  mutate(
    pass_rate = map_dbl(metric_label, function(m) {
      mean(quality_long$score[quality_long$metric_label == m] >= 2)
    }),
    # Annotation label: mean (pass%)
    label = sprintf("%.2f\n(%d%%)", mean, round(pass_rate * 100)),
    # Flag safety metrics for different fill
    metric_type = if_else(
      metric_label %in% c("Hallucination", "Harmfulness"),
      "Safety", "Quality"
    )
  )

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(kpi, aes(x = metric_label, y = mean, fill = metric_type)) +
  # Reference lines
  geom_hline(yintercept = 2,   colour = RED_SOFT,   linetype = "dashed", linewidth = 0.4) +
  geom_hline(yintercept = 2.5, colour = GRAY_MID,   linetype = "dotted", linewidth = 0.3) +
  geom_hline(yintercept = 3,   colour = GREEN_SOFT, linetype = "dotted", linewidth = 0.3) +
  # Bars
  geom_col(width = 0.6, colour = NA, alpha = 0.88) +
  # Bootstrap CI error bars
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.18, linewidth = 0.55, colour = TEAL_DARK) +
  # Mean + pass-rate label inside bar
  geom_text(aes(y = ci_lo - 0.06, label = label),
            vjust = 1, size = 2.4, colour = TEAL_DARK,
            fontface = "bold", lineheight = 0.9) +
  # Axis
  scale_y_continuous(
    limits = c(0, 3.25),
    breaks = 0:3,
    labels = c("0\nFail", "1\nPoor", "2\nAdequate", "3\nExcellent"),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  scale_fill_manual(
    values = c("Quality" = TEAL_MED, "Safety" = GOLD),
    name   = NULL,
    labels = c("Quality metrics", "Safety metrics")
  ) +
  # Threshold annotation
  annotate("text", x = 0.45, y = 2.03, label = "Pass threshold",
           hjust = 0, vjust = 0, size = 2.2, colour = RED_SOFT,
           fontface = "italic") +
  labs(
    title    = "Figure 1 — Quality Scorecard",
    subtitle = "Mean score ± 95% bootstrap CI · n = 42 ratings (21 items × 2 raters)",
    x        = NULL,
    y        = "Mean Quality Score (0–3)",
    caption  = "Bar labels: mean (pass rate ≥ 2).  Dashed line = pass threshold."
  ) +
  theme_poster() +
  theme(legend.position = "top")

# ── Export ────────────────────────────────────────────────────────────────────
ggsave("fig1_kpi_summary.pdf", plot = p,
       width = 3.5, height = 2.8, units = "in", device = cairo_pdf)
ggsave("fig1_kpi_summary.png", plot = p,
       width = 3.5, height = 2.8, units = "in", dpi = 300)

message("Saved fig1_kpi_summary.pdf + .png")
