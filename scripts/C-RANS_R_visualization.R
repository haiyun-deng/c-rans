# ==============================================================================
# R Script for C-RANS Benchmark 
# ==============================================================================

# 1. Load required packages
library(tidyverse)
library(ggsci)
library(extrafont)
library(tidytext)
library(magick)
library(scales)

# Load system fonts (ensure Helvetica/Arial is available)
loadfonts(device = "pdf", quiet = TRUE)

# 2. Load and prepare dataset
df_raw <- read.csv("benchmark_metrics.csv")

# Standardize column names
df <- df_raw %>% 
  rename(model = model_name) %>%
  mutate(
    log10_ppl = log10(eap_ppl),
    register = factor(register, 
                      levels = c("ExamEssay", "FreeWriting", "OralExam", "OralPractice"),
                      labels = c("a  ExamEssay", "b  FreeWriting", "c  OralExam", "d  OralPractice"))
  )

# Define universal Nature-style theme
theme_nature <- function() {
  theme_classic(base_size = 10, base_family = "Helvetica") +
    theme(
      text = element_text(size = 10),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      axis.text.x = element_text(angle = 35, hjust = 1),
      strip.text = element_text(size = 11, face = "bold", hjust = 0),
      strip.background = element_blank(),
      legend.position = "top",
      legend.title = element_blank()
    )
}

# 3. Figure 1: Grouped Bar Plot (Task A & B)
df_summary <- df %>%
  group_by(model) %>%
  summarise(
    a_zeroshot = mean(a_qwk_zeroshot, na.rm = TRUE),
    a_fewshot  = mean(a_qwk_fewshot, na.rm = TRUE),
    b_zeroshot = mean(b_qwk_zeroshot, na.rm = TRUE),
    b_fewshot  = mean(b_qwk_fewshot, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(-model, names_to = c("Task", "Setting"), names_pattern = "(.*)_(.*)", values_to = "QWK") %>%
  mutate(
    Task = ifelse(Task == "a", "a  Task A: Grammar Rating", "b  Task B: Naturalness Rating"),
    Setting = factor(ifelse(Setting == "zeroshot", "Zero-shot", "Few-shot"), levels = c("Zero-shot", "Few-shot"))
  ) %>%
  group_by(Task, model) %>%
  mutate(sort_val = max(QWK)) %>%
  ungroup()

p1 <- ggplot(df_summary, aes(x = reorder_within(model, sort_val, Task), y = QWK, fill = Setting)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.75, color = "black", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.3f", QWK)), position = position_dodge(width = 0.8), vjust = -0.5, size = 2.8) +
  facet_wrap(~ Task, nrow = 2, scales = "free_x") +
  scale_x_reordered() +
  scale_fill_npg() + # Elegant professional palette
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "Model (Ranked by Performance)", y = "Quadratic Weighted Kappa (QWK)") +
  theme_nature()

ggsave("Figure1_Barplot.pdf", plot = p1, width = 8.5, height = 7)

# 4. Figure 2-4: Box Plots for Distribution Analysis
plot_box <- function(data, y_var, y_lab, higher_better = TRUE) {
  data <- data %>% group_by(register, model) %>% 
    mutate(sort_val = mean(.data[[y_var]], na.rm = TRUE) * (if(higher_better) 1 else -1)) %>% ungroup()
  
  ggplot(data, aes(x = reorder_within(model, sort_val, register), y = .data[[y_var]], fill = register)) +
    geom_boxplot(width = 0.72, linewidth = 0.35, alpha = 0.8) +
    geom_jitter(aes(color = register), width = 0.1, size = 0.4, alpha = 0.15, show.legend = FALSE) +
    facet_wrap(~ register, nrow = 2, scales = "free_x") +
    scale_x_reordered() + scale_fill_npg() + scale_color_npg() +
    labs(x = "Model", y = y_lab) + theme_nature() + theme(legend.position = "none")
}

p2 <- plot_box(df, "GLEU", "GLEU Score", TRUE)
p3 <- plot_box(df, "cosine_similarity", "Cosine Similarity", TRUE)
p4a <- plot_box(df, "log10_ppl", "log10(PPL)", FALSE)
p4b <- plot_box(df, "NCD", "NCD", FALSE)

ggsave("Figure2_GLEU.pdf", p2, width = 8.5, height = 6)
ggsave("Figure3_Cosine.pdf", p3, width = 8.5, height = 6)
ggsave("Figure4a_PPL.pdf", p4a, width = 8.5, height = 6)
ggsave("Figure4b_NCD.pdf", p4b, width = 8.5, height = 6)

# 5. Figure 5: Model Performance Overview Heatmap
df_heatmap <- df %>%
  filter(!is.na(GLEU), !is.na(cosine_similarity), !is.na(log10_ppl), !is.na(NCD)) %>%
  group_by(model, register) %>%
  summarise(GLEU = median(GLEU), Cosine = median(cosine_similarity), PPL = median(log10_ppl), NCD = median(NCD), .groups = "drop") %>%
  group_by(register) %>%
  mutate(across(c(GLEU, Cosine), ~min_rank(desc(.)), .names = "rank_{.col}"),
         across(c(PPL, NCD), ~min_rank(.), .names = "rank_{.col}")) %>%
  group_by(model) %>%
  summarise(across(starts_with("rank_"), mean, .names = "{str_remove(.col, 'rank_')}"), .groups = "drop") %>%
  rowwise() %>% mutate(Overall = mean(c(GLEU, Cosine, PPL, NCD))) %>% ungroup()

df_plot_h <- df_heatmap %>%
  pivot_longer(-model, names_to = "Metric", values_to = "Rank") %>%
  mutate(model = factor(model, levels = rev(df_heatmap$model[order(df_heatmap$Overall)])),
         Metric = factor(Metric, levels = c("GLEU", "Cosine", "PPL", "NCD", "Overall")))

p5 <- ggplot(df_plot_h, aes(x = Metric, y = model, fill = Rank)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", Rank), color = Rank < 3.5), size = 3.5, fontface = "bold") +
  scale_fill_gradient(low = "#3C5488", high = "#F5F5F5") +
  scale_color_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
  labs(x = "Metric", y = "Model", fill = "Mean Rank") +
  theme_minimal(base_size = 10) + theme(panel.grid = element_blank(), axis.text = element_text(color = "black"))

ggsave("Figure5_Heatmap.pdf", p5, width = 7, height = 5)

# Convert all PDFs to high-resolution TIFF for submission
convert_to_tiff <- function(filename) {
  img <- image_read_pdf(filename, density = 600)
  image_write(img, path = str_replace(filename, ".pdf", ".tiff"), format = "tiff", compression = "lzw")
}
lapply(list.files(pattern = "Figure.*\\.pdf$"), convert_to_tiff)

# Supplementary Analysis: Calculate LLM-human correlation w.r.t PPL
# 1. Load data
df <- read.csv("PPL_vs_human_rating.csv")

# 2. Correlation test (Spearman)
cor_test <- cor.test(df$eap_ppl, df$human_rating, method = "spearman")
print(cor_test)

# Result:
Spearman's rank correlation rho

data:  df$eap_ppl and df$human_rating
S = 20325284, p-value < 2.2e-16
alternative hypothesis: true rho is not equal to 0
sample estimates:
       rho 
-0.6460479 
#