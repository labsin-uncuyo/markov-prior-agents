#!/usr/bin/env Rscript
# Journal-ready comparison of actions per episode

suppressMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
})

find_input_files <- function(data_dir, use_old = FALSE) {
  preferred_files <- if (use_old) {
    c("gemma3_4b.json", "MC_gemma3_4b.json")
  } else {
    c("episode_data_gemma3_4b.json", "episode_data_MC.json")
  }
  available_files <- list.files(data_dir, pattern = "\\.json$", full.names = FALSE)
  matched_files <- preferred_files[preferred_files %in% available_files]

  if (length(matched_files) == length(preferred_files)) {
    return(file.path(data_dir, matched_files))
  }

  character(0)
}

make_display_name <- function(file_name) {
  if (file_name %in% c("episode_data_gemma3_4b.json", "gemma3_4b.json")) {
    return("baseline_gemma3:4b")
  }
  if (file_name %in% c("episode_data_MC.json", "MC_gemma3_4b.json")) {
    return("MC_gemma3:4b")
  }
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
    stop("Usage: Rscript plot_episode_lengths.R [current|old]")
  }

  mode
}

extract_episode_lengths <- function(filepath) {
  data <- jsonlite::fromJSON(filepath, simplifyDataFrame = FALSE)
  file_name <- basename(filepath)
  display_name <- make_display_name(file_name)

  do.call(rbind, lapply(seq_along(data), function(idx) {
    episode <- data[[idx]]
    data.frame(
      source_file = file_name,
      display_file = display_name,
      episode = idx,
      action_count = length(episode$evaluation %||% list()),
      end_reason = if (is.null(episode$end_reason)) NA_character_ else episode$end_reason,
      stringsAsFactors = FALSE
    )
  }))
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

summarize_lengths <- function(lengths_df) {
  lengths_df %>%
    group_by(display_file) %>%
    summarize(
      n = n(),
      mean_actions = mean(action_count),
      sd_actions = sd(action_count),
      se_actions = sd_actions / sqrt(n),
      t_crit = qt(0.975, df = n - 1),
      ci_half_width = ifelse(is.na(se_actions), 0, t_crit * se_actions),
      ymin = mean_actions - ci_half_width,
      ymax = mean_actions + ci_half_width,
      label = sprintf("%.1f", mean_actions),
      .groups = "drop"
    )
}

build_plot <- function(lengths_df, summary_df, title_suffix) {
  ggplot(lengths_df, aes(x = display_file, y = action_count, color = display_file)) +
    geom_jitter(
      width = 0.08,
      height = 0,
      alpha = 0.7,
      size = 2.1,
      stroke = 0
    ) +
    geom_errorbar(
      data = summary_df,
      aes(x = display_file, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      width = 0.12,
      linewidth = 0.55,
      color = "black"
    ) +
    geom_point(
      data = summary_df,
      aes(x = display_file, y = mean_actions),
      inherit.aes = FALSE,
      shape = 21,
      size = 4.2,
      stroke = 0.55,
      fill = "white",
      color = "black"
    ) +
    geom_text(
      data = summary_df,
      aes(x = display_file, y = ymax + 4, label = label),
      inherit.aes = FALSE,
      size = 3.7,
      family = "sans"
    ) +
    scale_color_manual(
      values = c(
        "baseline_gemma3:4b" = "#4C78A8",
        "MC_gemma3:4b" = "#F58518"
      ),
      guide = "none"
    ) +
    scale_y_continuous(
      name = "Actions per episode",
      limits = c(0, 110),
      breaks = seq(0, 100, by = 20),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      x = NULL,
      title = sprintf("Average number of actions per episode, %s", title_suffix),
      subtitle = "Points show individual episodes; markers and bars show mean and 95% CI"
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(
        size = 14,
        face = "bold",
        hjust = 0.5,
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = 10.5,
        hjust = 0.5,
        margin = margin(b = 12),
        color = "#4D4D4D"
      ),
      axis.title.y = element_text(size = 12, margin = margin(r = 10)),
      axis.text.x = element_text(size = 11),
      axis.text.y = element_text(size = 10.5),
      axis.ticks = element_blank(),
      panel.grid.major.y = element_line(color = "#D9D9D9", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_rect(color = "#4D4D4D", fill = NA, linewidth = 0.5),
      plot.margin = margin(14, 18, 14, 14)
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
  output_stem <- if (use_old) {
    "episode_length_comparison_old"
  } else {
    "episode_length_comparison"
  }
  title_suffix <- if (use_old) "old dataset" else "current dataset"

  json_files <- find_input_files(data_dir, use_old = use_old)
  if (length(json_files) == 0) {
    cat(sprintf("Expected JSON files not found in %s.\n", data_dir))
    return()
  }

  lengths_df <- do.call(rbind, lapply(json_files, extract_episode_lengths))
  lengths_df$display_file <- factor(
    lengths_df$display_file,
    levels = c("baseline_gemma3:4b", "MC_gemma3:4b")
  )
  summary_df <- summarize_lengths(lengths_df)
  summary_df$display_file <- factor(
    summary_df$display_file,
    levels = c("baseline_gemma3:4b", "MC_gemma3:4b")
  )

  cat(sprintf(
    "Analyzed %d JSON file(s) with %d episodes.\n",
    length(json_files),
    nrow(lengths_df)
  ))
  print(summary_df)

  p <- build_plot(lengths_df, summary_df, title_suffix)

  ggsave(
    file.path(data_dir, sprintf("%s.png", output_stem)),
    plot = p,
    width = 7.0,
    height = 4.8,
    dpi = 300,
    bg = "white"
  )

  ggsave(
    file.path(data_dir, sprintf("%s.pdf", output_stem)),
    plot = p,
    width = 7.0,
    height = 4.8,
    bg = "white"
  )

  cat("\nPlot saved as:\n")
  cat(sprintf("  PNG: %s\n", file.path(data_dir, sprintf("%s.png", output_stem))))
  cat(sprintf("  PDF: %s\n", file.path(data_dir, sprintf("%s.pdf", output_stem))))
}

main()
