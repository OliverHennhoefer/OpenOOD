library(arrow)
library(ggplot2)

scan <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/1_scan.parquet")
disc <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/2_discr.parquet")
resl <- arrow::read_parquet("C:/Users/heol/Projects/OpenOOD/3_result.parquet")

scan$label <- as.factor(scan$label)
coi <- scan$layer4.1.conv2_layer_output_avg_317
ggplot(scan, aes(x = coi, fill = label)) +
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
