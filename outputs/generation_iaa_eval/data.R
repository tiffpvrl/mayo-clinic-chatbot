# ─────────────────────────────────────────────────────────────────────────────
# data.R  ·  Shared data prep + poster theme
# Source this file at the top of every figure script:
#   source("data.R")
# ─────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(ggplot2)

# ── Raw data (inline — no external file dependency) ───────────────────────────
raw <- tribble(
  ~clinician, ~patient_id,  ~turn, ~factual_q, ~accuracy_q, ~relevance_q, ~hallucination_q, ~harmfulness_q, ~factual, ~accurate, ~relevant, ~hallucination, ~harmful,
  "NM", "P823667513", 1, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P823667513", 2, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P823667513", 3, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P410324840", 1, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P410324840", 2, 2, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P410324840", 3, 3, 3, 3, 3, 2, "yes","yes","yes","no","no",
  "NM", "P194129405", 1, 2, 2, 3, 2, 2, "yes","yes","yes","no","no",
  "NM", "P194129405", 2, 2, 2, 2, 2, 2, "yes","yes","yes","no","no",
  "NM", "P194129405", 3, 3, 2, 3, 2, 3, "yes","yes","yes","no","no",
  "NM", "P444721444", 1, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P444721444", 2, 2, 2, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P444721444", 3, 2, 2, 2, 2, 2, "yes","no","no","no","no",
  "NM", "P820234749", 1, 2, 2, 2, 2, 2, "yes","yes","yes","no","no",
  "NM", "P820234749", 2, 2, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P820234749", 3, 2, 2, 2, 2, 1, "no","no","no","yes","yes",
  "NM", "P919567391", 1, 2, 2, 3, 2, 2, "yes","yes","yes","no","no",
  "NM", "P919567391", 2, 2, 2, 2, 2, 2, "yes","yes","yes","no","no",
  "NM", "P919567391", 3, 2, 2, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P404029370", 1, 3, 2, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P404029370", 2, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "NM", "P404029370", 3, 2, 2, 2, 2, 2, "yes","no","yes","no","yes",
  "PM", "P823667513", 1, 3, 2, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P823667513", 2, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P823667513", 3, 2, 3, 2, 2, 3, "yes","yes","yes","no","no",
  "PM", "P410324840", 1, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P410324840", 2, 2, 2, 2, 2, 2, "yes","yes","no","no","no",
  "PM", "P410324840", 3, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P194129405", 1, 2, 2, 1, 2, 2, "yes","no","no","yes","no",
  "PM", "P194129405", 2, 2, 2, 2, 2, 2, "yes","no","yes","no","no",
  "PM", "P194129405", 3, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P444721444", 1, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P444721444", 2, 2, 2, 3, 2, 3, "yes","yes","yes","no","no",
  "PM", "P444721444", 3, 2, 2, 1, 2, 2, "yes","no","no","yes","no",
  "PM", "P820234749", 1, 3, 2, 2, 2, 2, "yes","yes","yes","no","no",
  "PM", "P820234749", 2, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P820234749", 3, 2, 2, 1, 2, 1, "no","no","no","yes","yes",
  "PM", "P919567391", 1, 2, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P919567391", 2, 3, 2, 2, 2, 2, "yes","yes","yes","no","no",
  "PM", "P919567391", 3, 2, 2, 2, 3, 3, "yes","yes","yes","no","no",
  "PM", "P404029370", 1, 3, 2, 2, 3, 3, "yes","yes","yes","no","no",
  "PM", "P404029370", 2, 3, 3, 3, 3, 3, "yes","yes","yes","no","no",
  "PM", "P404029370", 3, 2, 2, 2, 3, 2, "yes","no","no","no","yes",
)

# ── Derived columns ────────────────────────────────────────────────────────────
raw <- raw |>
  mutate(
    turn_label = paste0("Turn ", turn),
    factual_bin       = if_else(factual       == "yes", 1L, 0L),
    accurate_bin      = if_else(accurate      == "yes", 1L, 0L),
    relevant_bin      = if_else(relevant      == "yes", 1L, 0L),
    hallucination_bin = if_else(hallucination == "yes", 1L, 0L),
    harmful_bin       = if_else(harmful       == "yes", 1L, 0L),
  )

# ── Tidy quality scores (long form) ───────────────────────────────────────────
quality_long <- raw |>
  pivot_longer(
    cols      = c(factual_q, accuracy_q, relevance_q, hallucination_q, harmfulness_q),
    names_to  = "metric",
    values_to = "score"
  ) |>
  mutate(
    metric_label = recode(metric,
      factual_q       = "Factual",
      accuracy_q      = "Accuracy",
      relevance_q     = "Relevance",
      hallucination_q = "Hallucination",
      harmfulness_q   = "Harmfulness"
    ),
    # Ordered factor: safety metrics last so they group naturally
    metric_label = factor(metric_label,
      levels = c("Factual", "Accuracy", "Relevance", "Hallucination", "Harmfulness"))
  )

# ── Poster theme ───────────────────────────────────────────────────────────────
# Designed for a ~3.5" wide × 2.5" tall panel at 300 dpi.
# Adjust base_size if your poster column is wider/narrower.

TEAL_DARK  <- "#1A4A5C"
TEAL_MED   <- "#2E7D9A"
TEAL_LIGHT <- "#D4EBF5"
GOLD       <- "#E09F1E"
RED_SOFT   <- "#C0392B"
GREEN_SOFT <- "#27AE60"
GRAY_MID   <- "#7F8C8D"
GRAY_LIGHT <- "#ECF0F1"

# Rater palette: NM = teal, PM = gold
rater_colors <- c("NM" = TEAL_DARK, "PM" = GOLD)
rater_fills  <- c("NM" = TEAL_MED,  "PM" = GOLD)

# Turn palette: darkens turn-by-turn
turn_colors <- c("Turn 1" = "#2E7D9A", "Turn 2" = "#1A4A5C", "Turn 3" = "#0D2530")

theme_poster <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
  theme(
    # Panel
    panel.background  = element_rect(fill = "white", colour = NA),
    panel.grid.major  = element_line(colour = GRAY_LIGHT, linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),

    # Axes
    axis.line         = element_line(colour = TEAL_DARK, linewidth = 0.4),
    axis.ticks        = element_line(colour = TEAL_DARK, linewidth = 0.3),
    axis.ticks.length = unit(2, "pt"),
    axis.text         = element_text(colour = TEAL_DARK, size = base_size - 1),
    axis.title        = element_text(colour = TEAL_DARK, size = base_size,
                                     face = "bold", margin = margin(t = 3, r = 3)),

    # Title / subtitle / caption
    plot.title    = element_text(colour = TEAL_DARK, size = base_size + 1,
                                 face = "bold", margin = margin(b = 3)),
    plot.subtitle = element_text(colour = GRAY_MID, size = base_size - 1,
                                 margin = margin(b = 4)),
    plot.caption  = element_text(colour = GRAY_MID, size = base_size - 2,
                                 hjust = 0, margin = margin(t = 4)),
    plot.margin   = margin(6, 8, 4, 6),
    plot.background = element_rect(fill = "white", colour = NA),

    # Legend
    legend.position    = "top",
    legend.direction   = "horizontal",
    legend.title       = element_text(colour = TEAL_DARK, size = base_size - 1,
                                      face = "bold"),
    legend.text        = element_text(colour = TEAL_DARK, size = base_size - 1),
    legend.key.size    = unit(8, "pt"),
    legend.margin      = margin(0, 0, 2, 0),
    legend.box.spacing = unit(2, "pt"),

    # Strip (facets)
    strip.background = element_rect(fill = TEAL_LIGHT, colour = NA),
    strip.text       = element_text(colour = TEAL_DARK, size = base_size - 1,
                                    face = "bold", margin = margin(3, 3, 3, 3)),
  )
}

# ── Bootstrap CI helper ────────────────────────────────────────────────────────
boot_ci <- function(x, n_boot = 2000, conf = 0.95) {
  set.seed(42)
  boots <- replicate(n_boot, mean(sample(x, length(x), replace = TRUE)))
  tibble(
    mean = mean(x),
    ci_lo = quantile(boots, (1 - conf) / 2),
    ci_hi = quantile(boots, 1 - (1 - conf) / 2)
  )
}

message("data.R loaded — raw (", nrow(raw), " rows), quality_long, theme_poster(), boot_ci() ready.")
