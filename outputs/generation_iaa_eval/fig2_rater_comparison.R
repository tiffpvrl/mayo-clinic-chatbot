# ─────────────────────────────────────────────────────────────────────────────
# fig2_rater_comparison.R  ·  NM vs PM mean score per metric (dot + CI)
#
# Output: fig2_rater_comparison.pdf  (3.5 × 2.6 in)
#
# Uses a dot plot rather than grouped bars — cleaner for a poster, and the
# connecting segment makes the direction of disagreement immediately visible.
# ─────────────────────────────────────────────────────────────────────────────

source("data.R")

# ── Per-rater CI ──────────────────────────────────────────────────────────────
rater_ci <- quality_long |>
  group_by(metric_label, clinician) |>
  group_modify(~ boot_ci(.x$score)) |>
  ungroup()

# Segment endpoints (one row per metric, NM mean vs PM mean)
seg_data <- rater_ci |>
  select(metric_label, clinician, mean) |>
  pivot_wider(names_from = clinician, values_from = mean) |>
  rename(nm_mean = NM, pm_mean = PM)

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(rater_ci, aes(x = mean, y = fct_rev(metric_label))) +
  # Horizontal reference at pass threshold
  geom_vline(xintercept = 2, colour = RED_SOFT, linetype = "dashed",
             linewidth = 0.4) +
  # Connecting segment between NM and PM means
  geom_segment(
    data = seg_data,
    aes(x = nm_mean, xend = pm_mean,
        y = fct_rev(metric_label), yend = fct_rev(metric_label)),
    colour = GRAY_MID, linewidth = 0.5, inherit.aes = FALSE
  ) +
  # CI error bars
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi, colour = clinician),
                 height = 0.18, linewidth = 0.5) +
  # Dots
  geom_point(aes(colour = clinician, shape = clinician), size = 2.6) +
  # Numeric labels offset above dots
  geom_text(aes(label = sprintf("%.2f", mean), colour = clinician),
            nudge_y = 0.28, size = 2.2, fontface = "bold") +
  scale_x_continuous(
    limits = c(1.8, 3.1),
    breaks = c(2, 2.5, 3),
    labels = c("2\n(Adequate)", "2.5", "3\n(Excellent)")
  ) +
  scale_colour_manual(values = rater_colors, name = "Rater") +
  scale_shape_manual(values = c("NM" = 16, "PM" = 17), name = "Rater") +
  # Threshold label
  annotate("text", x = 2.01, y = 0.55, label = "Pass\nthreshold",
           hjust = 0, size = 2.0, colour = RED_SOFT, fontface = "italic",
           lineheight = 0.9) +
  labs(
    title    = "Figure 2 — Rater Severity Comparison",
    subtitle = "Mean ± 95% bootstrap CI per rater · NM vs PM · n = 21 ratings each",
    x        = "Mean Score (0–3)",
    y        = NULL,
    caption  = "Segments connect NM and PM means. Largest gap: Relevance (NM 2.71 vs PM 2.33)."
  ) +
  theme_poster() +
  theme(
    legend.position = "top",
    axis.line.y     = element_blank(),
    axis.ticks.y    = element_blank(),
    panel.grid.major.y = element_line(colour = GRAY_LIGHT, linewidth = 0.3),
    panel.grid.major.x = element_blank(),
  )

# ── Export ────────────────────────────────────────────────────────────────────
ggsave("fig2_rater_comparison.pdf", plot = p,
       width = 3.5, height = 2.6, units = "in", device = cairo_pdf)
ggsave("fig2_rater_comparison.png", plot = p,
       width = 3.5, height = 2.6, units = "in", dpi = 300)

message("Saved fig2_rater_comparison.pdf + .png")
