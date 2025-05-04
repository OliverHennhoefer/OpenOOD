library(arrow)
library(ggplot2)
library(dplyr)
library(hrbrthemes)  # For better typography
library(scales)      # For better formatting of axis values
library(tikzDevice)
library(tinytex)


set.seed(123)
setwd(tcltk::tk_choose.dir())  # Choose Project Root for OpenOOD (commented for reproducibility)

# ------------------------------------------------------------------------------

# Load data
col_discr <- arrow::read_parquet('2_discr.parquet')
# Convert to regular data frame to avoid arrow-specific subsetting issues
col_discr <- as.data.frame(col_discr)
col_discr <- col_discr[order(col_discr$average_divergence, decreasing = TRUE), ]

# Add percentile ranking for reference
col_discr <- col_discr %>%
  mutate(percentile = row_number() / n() * 100,
         feature_index = row_number())  # Add explicit index for plotting

# Select points to highlight (every 10% of data)
highlight_indices <- floor(seq(1, nrow(col_discr), length.out = 10))
highlight_points <- col_discr[highlight_indices, ]

# Create a publication-ready plot
(p <- ggplot() +
  # Main line with better color and size
  geom_line(data = col_discr,
            aes(x = feature_index, y = average_divergence),
            color = "#3182bd", 
            linewidth = 1.2) +
  # Better labeling
  labs(
    x = "Network Representation Features (ranked)",
    y = "Avg. Hellinger Distance Between Class Distributions",
  ) +
  # Create a secondary x-axis showing percentiles
  scale_x_continuous(
    breaks = pretty(1:nrow(col_discr), n = 6)
  ) +
  # Adjust y-axis range to focus on the relevant data range
  scale_y_continuous(
    limits = function(x) {
      data_range <- range(col_discr$average_divergence)
      min_y <- data_range[1] * 0.975  # 2.5% below minimum but not below 0
      if(min_y < 0) min_y <- 0
      max_y <- data_range[2] * 1.025  # 2.5% above maximum
      return(c(min_y, max_y))
    }
  ) +
  # Apply a clean theme with better typography
  theme_minimal() +
  # Custom theme elements
  theme(
    axis.title = element_text(size = 12, face = "bold", margin = margin(t = 10, b = 10)),
    axis.text = element_text(size = 12, color = "black"),
    axis.line = element_line(color = "black", linewidth = 0.5),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    axis.ticks.length = unit(0.2, "cm"),
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5)
  ))


options(tikzLatex = "C:/Users/heol/AppData/Roaming/TinyTeX/bin/windows/pdflatex.exe")
options(tikzLatexPackages = c(
  "\\usepackage[utf8]{inputenc}",
  "\\usepackage[T1]{fontenc}",
  "\\usepackage{lmodern}",
  "\\usepackage{tikz}"
))

library(tikzDevice)
tikz("./results/visu/discriminative_feature.tex", width = 7, height = 5, standAlone = TRUE)
print(p)
dev.off()

# ------------------------------------------------------------------------------


