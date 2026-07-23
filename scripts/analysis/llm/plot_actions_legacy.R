#!/usr/bin/env Rscript
# Journal-ready grouped bar plot for action evaluation outcomes

suppressMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(scales)
})

categorize_score <- function(score) {
  if (score %in% c(8, 10)) return("Good")
  if (score == 0) return("Bad")
  return("Boring")
}

analyze_json_file <- function(filepath) {
  data <- jsonlite::fromJSON(filepath, simplifyDataFrame = FALSE)
  file_name <- basename(filepath)

  scores_list <- list()
  score_idx <- 1

  for (episode in data) {
    if (!is.null(episode$evaluation)) {
      for (score in episode$evaluation) {
        scores_list[[score_idx]] <- data.frame(
          score = score,
          category = categorize_score(score),
          file = file_name,
          stringsAsFactors = FALSE
        )
        score_idx <- score_idx + 1
      }
    }
  }

  do.call(rbind, scores_list)
}

find_input_files <- function(data_dir, use_old = FALSE) {
  preferred_files <- if (use_old) {
    c("gemma4b.json", "MC_gemma4b.json")
  } else {
    c("episode_data_gemma4b.json", "episode_data_MC.json")
  }
  available_files <- list.files(data_dir, pattern = "\\.json$", full.names = FALSE)
  matched_files <- preferred_files[preferred_files %in% available_files]

  if (length(matched_files) == length(preferred_files)) {
    return(file.path(data_dir, matched_files))
  }

  character(0)
}

make_display_name <- function(file_name) {
  if (file_name == "episode_data_gemma4b.json") return("LLM Baseline")
  if (file_name == "episode_data_MC.json") return("LLM-MC")
  if (file_name == "gemma4b.json") return("LLM Baseline")
  if (file_name == "MC_gemma4b.json") return("LLM-MC")
  tools::file_path_sans_ext(file_name)
}

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

get_run_mode <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) == 0) {
    return("current")
  }

  mode <- tolower(args[1])
  if (!mode %in% c("current", "old")) {
    stop("Usage: Rscript plot_actions.R [current|old]")
  }

  mode
}

build_plot <- function(counts) {
  dodge_width <- 0.99
  category_display <- c(
    "Bad" = "Invalid",
    "Boring" = "Redundant",
    "Good" = "Effective"
  )

  # Palette adjusted to match the reference example more closely:
  # strong green for "Good", vivid orange for "Boring", and a muted
  # neutral/warning tone for "Bad" so it remains distinct without
  # overpowering the chart.
  category_colors <- c(
    "Bad"    = "#C44E52",
    "Boring" = "#FF7F0E",
    "Good"   = "#2CA02C"
  )
  # Value labels inherit bar color so they stay legible without extra ink
  label_colors <- c(
    "Bad"    = "#8F383B",
    "Boring" = "#C65F00",
    "Good"   = "#1F7A1F"
  )

  # Direct category labels: annotate only above the first group (LLM Baseline).
  # The color mapping then carries the reader through the second group.
  baseline_labels <- counts[counts$display_file == "LLM Baseline", , drop = FALSE]

  ggplot(counts, aes(x = display_file, y = pct, fill = category)) +
    geom_col(
      position = position_dodge(width = dodge_width),
      width = 0.70,
      color = NA        # SWD: remove bar outlines — pure data ink
    ) +
    # Percentage values just above each bar
    geom_text(
      aes(label = label, color = category),
      position = position_dodge(width = dodge_width),
      vjust = -0.45,
      size = 2.7,
      family = "serif",
      fontface = "bold",
      show.legend = FALSE
    ) +
    # Category names above the LLM Baseline bars only — no legend needed
    geom_text(
      data = baseline_labels,
      aes(label = category_display[as.character(category)], color = category, y = pct),
      position = position_dodge(width = dodge_width),
      vjust = -2.5,     # sits above the % label
      size = 2.5,
      family = "serif",
      fontface = "italic",
      show.legend = FALSE
    ) +
    scale_fill_manual(values = category_colors,  guide = "none") +
    scale_color_manual(values = label_colors,     guide = "none") +
    # SWD: if bars carry their own labels, the y-axis is redundant — remove it
    scale_y_continuous(
      limits = c(0, 100),
      expand = expansion(mult = c(0, 0.22))   # headroom for category labels
    ) +
    labs(
      x    = NULL,
      y    = NULL,
      # SWD: title states the finding, not the topic; left-aligned
      title    = "Actions performance",
      subtitle = "Distribution of actions by outcome  |  Scenario 1"
    ) +
    coord_cartesian(clip = "off") +
    theme_classic(base_size = 10, base_family = "serif") +
    theme(
      # SWD: left-aligned title gives a natural reading entry point
      plot.title    = element_text(size = 11, face = "bold",  hjust = 0,
                                   margin = margin(b = 3)),
      plot.subtitle = element_text(size = 8.5, color = "gray45", hjust = 0,
                                   margin = margin(b = 10)),
      # Remove y-axis entirely — bars are labeled
      axis.text.y   = element_blank(),
      axis.line.y   = element_blank(),
      axis.ticks    = element_blank(),
      # Keep only the baseline x-axis line; group labels are the only annotation needed
      axis.line.x   = element_line(color = "gray40", linewidth = 0.4),
      axis.text.x   = element_text(size = 10, color = "black"),
      # No gridlines — they compete with the data and are redundant with labels
      panel.grid    = element_blank(),
      plot.margin   = margin(10, 16, 8, 8)
    )
}

main <- function() {
  script_dir <- get_script_dir()
  run_mode <- get_run_mode()
  use_old <- identical(run_mode, "old")
  data_dir <- if (use_old) {
    file.path(script_dir, "data", "old")
  } else {
    file.path(script_dir, "data")
  }
  output_dir <- data_dir
  output_stem <- if (use_old) {
    "action_categories_barplot_old"
  } else {
    "action_categories_barplot"
  }

  json_files <- find_input_files(data_dir, use_old = use_old)

  if (length(json_files) == 0) {
    cat(sprintf("Expected JSON files not found in %s.\n", data_dir))
    return()
  }

  all_data <- do.call(rbind, lapply(json_files, analyze_json_file))
  all_data <- data.frame(all_data, stringsAsFactors = FALSE)
  all_data$display_file <- vapply(all_data$file, make_display_name, character(1))

  counts <- all_data %>%
    group_by(display_file, category) %>%
    summarize(count = n(), .groups = "drop") %>%
    group_by(display_file) %>%
    mutate(pct = count / sum(count) * 100) %>%
    ungroup()

  counts$display_file <- factor(
    counts$display_file,
    levels = c("LLM Baseline", "LLM-MC")
  )
  counts$category <- factor(
    counts$category,
    levels = c("Bad", "Boring", "Good")
  )
  counts <- counts[order(counts$display_file, counts$category), ]
  counts$label <- sprintf("%.1f%%", counts$pct)
  counts$category_display <- c(
    "Bad" = "Invalid",
    "Boring" = "Redundant",
    "Good" = "Effective"
  )[as.character(counts$category)]

  cat(sprintf(
    "Analyzed %d JSON file(s) with %d total actions.\n",
    length(json_files),
    nrow(all_data)
  ))
  print(counts[, c("display_file", "category_display", "count", "pct", "label")])

  p <- build_plot(counts)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  ggsave(
    file.path(output_dir, sprintf("%s.png", output_stem)),
    plot = p,
    width = 3.5,
    height = 1.8,
    dpi = 300,
    bg = "white"
  )

  ggsave(
    file.path(output_dir, sprintf("%s.pdf", output_stem)),
    plot = p,
    device = cairo_pdf,
    width = 3.5,
    height = 1.8,
    bg = "white"
  )

  cat("\nBar plot saved as:\n")
  cat(sprintf("  PNG: %s\n", file.path(output_dir, sprintf("%s.png", output_stem))))
  cat(sprintf("  PDF: %s\n", file.path(output_dir, sprintf("%s.pdf", output_stem))))
}

main()
