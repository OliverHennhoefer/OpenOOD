library(arrow)

scan <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/1_scan.csv")
disc <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/2_disc.parquet")
resl <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/3_result.parquet")

coi <- scan$relu1_relevance_avg_4  #relu3_pre_activation_55  #relu3_pre_activation_11
ggplot(scan, aes(x = coi, fill = group)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 100) +
  facet_wrap(~ label, scales = "fixed") +  # Use free_y to adjust y-axis for each facet
  geom_vline(xintercept = quantile(coi, 0.1), linetype = "dashed", color = "black") +
  geom_vline(xintercept = quantile(coi, 0.5), linetype = "dashed", color = "black") +
  geom_vline(xintercept = quantile(coi, 0.9), linetype = "dashed", color = "black") +
  labs(title = "Comparison of Distributions Between Datasets by Group",
       x = "Value",
       y = "Count",
       fill = "Data Source") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")