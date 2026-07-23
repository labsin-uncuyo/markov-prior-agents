#!/usr/bin/env Rscript
# Plot LLM/ReAct action outcome distributions from processed summaries.

suppressMessages({
  library(ggplot2)
  library(dplyr)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  matched <- grep("--file=", args, value = TRUE)
  if (length(matched) > 0) {
    return(dirname(normalizePath(sub("--file=", "", matched[1]))))
  }
  normalizePath(getwd())
}

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "150_episodes"

script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
input_csv <- file.path(repo_root, "data", "processed", "llm", dataset, "action_outcome_summary.csv")
output_dir <- file.path(repo_root, "figures", "llm", dataset)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts <- read.csv(input_csv, stringsAsFactors = FALSE) %>%
  mutate(
    agent = factor(agent, levels = c("LLM Baseline", "LLM-MC")),
    category = factor(category, levels = c("Invalid", "Redundant", "Effective")),
    label = sprintf("%.1f%%", pct)
  )

colors <- c(
  "Invalid" = "#C44E52",
  "Redundant" = "#FF7F0E",
  "Effective" = "#2CA02C"
)

label_colors <- c(
  "Invalid" = "#8F383B",
  "Redundant" = "#C65F00",
  "Effective" = "#1F7A1F"
)

# Direct category labels above the LLM Baseline bars act as an inline legend,
# matching the original 30-episode figure style.
baseline_labels <- counts[counts$agent == "LLM Baseline", , drop = FALSE]

p <- ggplot(counts, aes(x = agent, y = pct, fill = category)) +
  geom_col(position = position_dodge(width = 0.99), width = 0.70, color = NA) +
  geom_text(
    aes(label = label, color = category),
    position = position_dodge(width = 0.99),
    vjust = -0.45,
    size = 2.7,
    family = "serif",
    fontface = "bold",
    show.legend = FALSE
  ) +
  geom_text(
    data = baseline_labels,
    aes(label = as.character(category), color = category, y = pct),
    position = position_dodge(width = 0.99),
    vjust = -2.5,
    size = 2.5,
    family = "serif",
    fontface = "italic",
    show.legend = FALSE
  ) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_color_manual(values = label_colors, guide = "none") +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.22))) +
  labs(
    x = NULL,
    y = NULL,
    title = "Actions performance",
    subtitle = sprintf("Distribution of actions by outcome  |  %s", gsub("_", " ", dataset))
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 10, base_family = "serif") +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0, margin = margin(b = 3)),
    plot.subtitle = element_text(size = 8.5, color = "gray45", hjust = 0, margin = margin(b = 10)),
    axis.text.y = element_blank(),
    axis.line.y = element_blank(),
    axis.ticks = element_blank(),
    axis.line.x = element_line(color = "gray40", linewidth = 0.4),
    axis.text.x = element_text(size = 10, color = "black"),
    panel.grid = element_blank(),
    plot.margin = margin(10, 16, 8, 8)
  )

ggsave(file.path(output_dir, "action_categories_barplot.png"), p, width = 3.5, height = 1.8, dpi = 300, bg = "white")
ggsave(file.path(output_dir, "action_categories_barplot.pdf"), p, width = 3.5, height = 1.8, bg = "white")
cat(sprintf("Saved LLM action plot to %s\n", output_dir))
