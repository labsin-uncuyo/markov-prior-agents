#!/usr/bin/env Rscript
# Journal-ready bar plot for action-level ALL vs BEPR outcomes

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  matched <- grep(file_arg, args, value = TRUE)

  if (length(matched) > 0) {
    script_path <- sub(file_arg, "", matched[1])
    return(dirname(normalizePath(script_path)))
  }

  normalizePath(getwd())
}

ensure_summary <- function(script_dir, data_dir) {
  summary_path <- file.path(data_dir, "best_episode_action_summary_markov_chain.csv")
  summarizer_path <- file.path(script_dir, "summarize_best_episode_actions_markov_chain.py")
  source_files <- file.path(
    data_dir,
    c(
      "parsed_populationGA.json",
      "parsed_populationRA10.json",
      "parsed_populationGPT.json",
      "parsed_populationED.json"
    )
  )

  needs_refresh <- !file.exists(summary_path) ||
    any(file.info(source_files)$mtime > file.info(summary_path)$mtime) ||
    file.info(summarizer_path)$mtime > file.info(summary_path)$mtime

  if (needs_refresh) {
    status <- system2(
      "python3",
      c(summarizer_path),
      stdout = TRUE,
      stderr = TRUE
    )

    if (!file.exists(summary_path)) {
      stop(paste(status, collapse = "\n"))
    }
  }

  summary_path
}

build_counts <- function(summary_path) {
  counts <- read.csv(summary_path, stringsAsFactors = FALSE) %>%
    mutate(
      panel = factor(panel, levels = c("ALL", "BEPR")),
      display_file = factor(
        display_file,
        levels = c(
          "Genetic Agent",
          "Random Agent",
          "GPT4o Agent",
          "Expert Defined (GPT o3-mini)"
        )
      ),
      category = factor(category, levels = c("Boring", "Good"))
    ) %>%
    arrange(panel, display_file, category) %>%
    group_by(panel, display_file) %>%
    mutate(
      pct_label = case_when(
        category == "Good" & pct >= 6 & display_file == "Genetic Agent" ~ sprintf("Good\n%.1f%%", pct),
        category == "Boring" & pct > 0 & display_file == "Genetic Agent" ~ sprintf("Boring\n%.1f%%", pct),
        category == "Good" & pct >= 6 ~ sprintf("%.1f%%", pct),
        category == "Boring" & pct > 0 ~ sprintf("%.1f%%", pct),
        TRUE ~ ""
      ),
      label_y = rev(cumsum(rev(pct)) - rev(pct) / 2),
      label_outside = category == "Boring" & pct < 8,
      label_y = ifelse(label_outside, 102.5, label_y)
    ) %>%
    ungroup()

  counts$category_display <- c(
    "Boring" = "Redundant",
    "Good" = "Effective"
  )[as.character(counts$category)]

  counts
}

build_plot <- function(counts) {
  category_colors <- c(
    "Boring" = "#FF7F0E",
    "Good" = "#1E7F2A"
  )
  label_data <- counts %>%
    filter(pct_label != "") %>%
    mutate(
      text_color = case_when(
        category == "Boring" & label_outside ~ category_colors["Boring"],
        category == "Good" ~ "white",
        TRUE ~ "#5C2000"
      )
    )
  panel_note <- data.frame(
    panel = factor(c("ALL", "BEPR"), levels = c("ALL", "BEPR")),
      x = 2.5,
      y = 108.0,
      label = c(
        "ALL uses all actions across all episodes",
        "BEPR uses all actions in the shortest successful episode per run"
    )
  )

  ggplot(counts, aes(x = display_file, y = pct, fill = category)) +
    geom_col(
      width = 0.78,
      color = NA
    ) +
    geom_text(
      data = label_data,
      aes(
        y = label_y,
        label = ifelse(
          display_file == "Genetic Agent",
          paste0(category_display, "\n", sub("^[^\n]+\n", "", pct_label)),
          pct_label
        ),
        color = text_color
      ),
      family = "serif",
      fontface = "bold",
      size = 2.9,
      lineheight = 0.9,
      vjust = ifelse(label_data$label_outside, 0, 0.5),
      show.legend = FALSE
    ) +
    geom_text(
      data = panel_note,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      family = "serif",
      size = 3.0,
      color = "gray35",
      fontface = "italic"
    ) +
    scale_fill_manual(values = category_colors, guide = "none") +
    scale_color_identity() +
    scale_y_continuous(
      limits = c(0, 111),
      labels = label_number(suffix = "%"),
      expand = expansion(mult = c(0, 0.02))
    ) +
    scale_x_discrete(
      labels = c(
        "Genetic Agent" = "Genetic",
        "Random Agent" = "Random",
        "GPT4o Agent" = "GPT4o",
        "Expert Defined (GPT o3-mini)" = "Expert Defined"
      )
    ) +
    facet_wrap(~panel, nrow = 1) +
    labs(
      x = NULL,
      y = "Percentage",
      title = "Actions performance",
      subtitle = "Distribution of actions by outcome  |  ALL vs best episode per run"
    ) +
    theme_classic(base_size = 10, base_family = "serif") +
    theme(
      plot.title = element_text(size = 21, face = "bold", hjust = 0,
                                margin = margin(b = 4)),
      plot.subtitle = element_text(size = 14, color = "gray45", hjust = 0,
                                   margin = margin(b = 14)),
      axis.title.y = element_text(size = 11),
      axis.text.x = element_text(size = 10.5, color = "black"),
      axis.text.y = element_text(size = 9.5, color = "gray25"),
      axis.line.x = element_blank(),
      axis.line.y = element_line(color = "gray45", linewidth = 0.5),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      legend.position = "none",
      strip.text = element_text(size = 18, face = "bold"),
      plot.margin = margin(14, 18, 8, 8)
    )
}

main <- function() {
  script_dir <- get_script_dir()
  data_dir <- file.path(script_dir, "data", "results_markov_chain")
  output_stem <- "best_episode_action_barplot_markov_chain"

  summary_path <- ensure_summary(script_dir, data_dir)
  counts <- build_counts(summary_path)

  cat("Loaded summary:\n")
  print(counts[, c("panel", "display_file", "category_display", "count", "pct")], row.names = FALSE)

  plot <- build_plot(counts)

  ggsave(
    file.path(data_dir, sprintf("%s.png", output_stem)),
    plot = plot,
    width = 10.4,
    height = 4.9,
    dpi = 300,
    bg = "white"
  )

  ggsave(
    file.path(data_dir, sprintf("%s.pdf", output_stem)),
    plot = plot,
    device = cairo_pdf,
    width = 10.4,
    height = 4.9,
    bg = "white"
  )

  cat("\nBar plot saved as:\n")
  cat(sprintf("  PNG: %s\n", file.path(data_dir, sprintf("%s.png", output_stem))))
  cat(sprintf("  PDF: %s\n", file.path(data_dir, sprintf("%s.pdf", output_stem))))
}

main()
