#!/usr/bin/env Rscript

# Export R circumplex test fixtures for Python regression testing
# This script generates JSON files containing expected outputs from R analyses
# to validate numerical parity in the Python port

library(circumplex)
library(jsonlite)

# Create output directory
output_dir <- "python_test_fixtures"
dir.create(output_dir, showWarnings = FALSE)

# Helper function to convert results to serializable format
prepare_results <- function(res) {
    list(
        results = as.data.frame(res$results),
        scores = as.data.frame(res$scores),
        details = res$details,
        type = res$type
    )
}

# Load example datasets
data("jz2017")
data("aw2009")

cat("Generating test fixtures for Python regression tests...\n\n")

# =============================================================================
# Test 1: Single-group mean-based SSM
# =============================================================================
cat("1. Single-group mean-based SSM (aw2009)...\n")
set.seed(12345)
res1 <- ssm_analyze(aw2009, scales = 1:8, boots = 2000, interval = 0.95)

# Extract key values for validation
fixture1 <- list(
    dataset = "aw2009",
    analysis_type = "single_group_mean",
    seed = 12345,
    input = list(
        scales = names(aw2009)[1:8],
        boots = 2000,
        interval = 0.95,
        listwise = TRUE
    ),
    expected = list(
        # Parameter estimates
        e_est = round(res1$results$e_est, 3),
        x_est = round(res1$results$x_est, 3),
        y_est = round(res1$results$y_est, 3),
        a_est = round(res1$results$a_est, 3),
        d_est = as.numeric(round(res1$results$d_est, 3)),
        fit_est = round(res1$results$fit_est, 3),
        # Confidence intervals
        e_lci = round(res1$results$e_lci, 3),
        e_uci = round(res1$results$e_uci, 3),
        x_lci = round(res1$results$x_lci, 3),
        x_uci = round(res1$results$x_uci, 3),
        y_lci = round(res1$results$y_lci, 3),
        y_uci = round(res1$results$y_uci, 3),
        a_lci = round(res1$results$a_lci, 3),
        a_uci = round(res1$results$a_uci, 3),
        d_lci = as.numeric(round(res1$results$d_lci, 3)),
        d_uci = as.numeric(round(res1$results$d_uci, 3)),
        # Scale scores
        scores = as.list(round(res1$scores[1, names(aw2009)[1:8]], 3))
    )
)
write_json(fixture1, file.path(output_dir, "ssm_single_group_mean.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 2: Multiple-group mean-based SSM
# =============================================================================
cat("2. Multiple-group mean-based SSM (jz2017)...\n")
set.seed(12345)
res2 <- ssm_analyze(jz2017, scales = 2:9, grouping = "Gender", boots = 2000)

fixture2 <- list(
    dataset = "jz2017",
    analysis_type = "multi_group_mean",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        grouping = "Gender",
        boots = 2000,
        interval = 0.95
    ),
    expected = list(
        # Parameter estimates (2 groups)
        e_est = round(res2$results$e_est, 3),
        x_est = round(res2$results$x_est, 3),
        y_est = round(res2$results$y_est, 3),
        a_est = round(res2$results$a_est, 3),
        d_est = as.numeric(round(res2$results$d_est, 3)),
        fit_est = round(res2$results$fit_est, 3),
        labels = as.character(res2$results$Label),
        # Confidence intervals
        e_lci = round(res2$results$e_lci, 3),
        e_uci = round(res2$results$e_uci, 3),
        x_lci = round(res2$results$x_lci, 3),
        x_uci = round(res2$results$x_uci, 3),
        y_lci = round(res2$results$y_lci, 3),
        y_uci = round(res2$results$y_uci, 3),
        a_lci = round(res2$results$a_lci, 3),
        a_uci = round(res2$results$a_uci, 3),
        d_lci = as.numeric(round(res2$results$d_lci, 3)),
        d_uci = as.numeric(round(res2$results$d_uci, 3)),
        # Scale scores (both groups)
        scores_female = as.list(round(res2$scores[1, names(jz2017)[2:9]], 3)),
        scores_male = as.list(round(res2$scores[2, names(jz2017)[2:9]], 3))
    )
)
write_json(fixture2, file.path(output_dir, "ssm_multi_group_mean.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 3: Multiple-group mean-based SSM with contrast
# =============================================================================
cat("3. Multiple-group mean-based SSM with contrast (jz2017)...\n")
set.seed(12345)
res3 <- ssm_analyze(jz2017,
    scales = 2:9, grouping = "Gender",
    contrast = TRUE, boots = 2000
)

fixture3 <- list(
    dataset = "jz2017",
    analysis_type = "multi_group_mean_contrast",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        grouping = "Gender",
        contrast = TRUE,
        boots = 2000
    ),
    expected = list(
        # All 3 rows: Female, Male, Male - Female
        e_est = round(res3$results$e_est, 3),
        x_est = round(res3$results$x_est, 3),
        y_est = round(res3$results$y_est, 3),
        a_est = round(res3$results$a_est, 3),
        d_est = as.numeric(round(res3$results$d_est, 3)),
        fit_est = round(res3$results$fit_est, 3),
        labels = as.character(res3$results$Label),
        # CIs for all rows
        e_lci = round(res3$results$e_lci, 3),
        e_uci = round(res3$results$e_uci, 3),
        a_lci = round(res3$results$a_lci, 3),
        a_uci = round(res3$results$a_uci, 3),
        d_lci = as.numeric(round(res3$results$d_lci, 3)),
        d_uci = as.numeric(round(res3$results$d_uci, 3))
    )
)
write_json(fixture3, file.path(output_dir, "ssm_multi_group_contrast.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 4: Single-group correlation-based SSM
# =============================================================================
cat("4. Single-group correlation-based SSM (jz2017)...\n")
set.seed(12345)
res4 <- ssm_analyze(jz2017, scales = 2:9, measures = "PARPD", boots = 2000)

fixture4 <- list(
    dataset = "jz2017",
    analysis_type = "single_group_correlation",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        measures = "PARPD",
        boots = 2000
    ),
    expected = list(
        e_est = round(res4$results$e_est, 3),
        x_est = round(res4$results$x_est, 3),
        y_est = round(res4$results$y_est, 3),
        a_est = round(res4$results$a_est, 3),
        d_est = as.numeric(round(res4$results$d_est, 1)),
        fit_est = round(res4$results$fit_est, 3),
        e_lci = round(res4$results$e_lci, 3),
        e_uci = round(res4$results$e_uci, 3),
        x_lci = round(res4$results$x_lci, 3),
        x_uci = round(res4$results$x_uci, 3),
        y_lci = round(res4$results$y_lci, 3),
        y_uci = round(res4$results$y_uci, 3)
    )
)
write_json(fixture4, file.path(output_dir, "ssm_single_group_correlation.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 5: Multi-group correlation-based SSM (no contrast)
# =============================================================================
cat("5. Multi-group correlation-based SSM (jz2017)...\n")
set.seed(12345)
res5 <- ssm_analyze(jz2017,
    scales = 2:9, measures = "PARPD",
    grouping = "Gender", boots = 2000
)

fixture5 <- list(
    dataset = "jz2017",
    analysis_type = "multi_group_correlation",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        measures = "PARPD",
        grouping = "Gender",
        boots = 2000
    ),
    expected = list(
        # Both groups
        e_est = round(res5$results$e_est, 3),
        x_est = round(res5$results$x_est, 3),
        y_est = round(res5$results$y_est, 3),
        a_est = round(res5$results$a_est, 3),
        d_est = as.numeric(round(res5$results$d_est, 1)),
        fit_est = round(res5$results$fit_est, 3),
        labels = as.character(res5$results$Label)
    )
)
write_json(fixture5, file.path(output_dir, "ssm_multi_group_correlation.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 6: Measure-contrast correlation-based SSM
# =============================================================================
cat("6. Measure-contrast correlation-based SSM (jz2017)...\n")
set.seed(12345)
res6 <- ssm_analyze(jz2017,
    scales = 2:9, measures = c("ASPD", "NARPD"),
    contrast = TRUE, boots = 2000
)

fixture6 <- list(
    dataset = "jz2017",
    analysis_type = "measure_contrast_correlation",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        measures = c("ASPD", "NARPD"),
        contrast = TRUE,
        boots = 2000
    ),
    expected = list(
        # All 3 rows: ASPD, NARPD, NARPD - ASPD
        e_est = round(res6$results$e_est, 3),
        x_est = round(res6$results$x_est, 3),
        y_est = round(res6$results$y_est, 3),
        a_est = round(res6$results$a_est, 3),
        d_est = as.numeric(round(res6$results$d_est, 1)),
        fit_est = round(res6$results$fit_est, 3),
        labels = as.character(res6$results$Label),
        # Some CIs
        e_lci = round(res6$results$e_lci, 3),
        e_uci = round(res6$results$e_uci, 3),
        a_lci = round(res6$results$a_lci, 3),
        a_uci = round(res6$results$a_uci, 3),
        # Scores for all three
        scores_aspd = as.list(round(res6$scores[1, names(jz2017)[2:9]], 3)),
        scores_narpd = as.list(round(res6$scores[2, names(jz2017)[2:9]], 3)),
        scores_contrast = as.list(round(res6$scores[3, names(jz2017)[2:9]], 3))
    )
)
write_json(fixture6, file.path(output_dir, "ssm_measure_contrast_correlation.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 7: Group-contrast correlation-based SSM
# =============================================================================
cat("7. Group-contrast correlation-based SSM (jz2017)...\n")
set.seed(12345)
res7 <- ssm_analyze(jz2017,
    scales = 2:9, measures = "NARPD",
    grouping = "Gender", contrast = TRUE, boots = 2000
)

fixture7 <- list(
    dataset = "jz2017",
    analysis_type = "group_contrast_correlation",
    seed = 12345,
    input = list(
        scales = names(jz2017)[2:9],
        measures = "NARPD",
        grouping = "Gender",
        contrast = TRUE,
        boots = 2000
    ),
    expected = list(
        # All 3 rows: Female, Male, Male - Female
        e_est = round(res7$results$e_est, 3),
        x_est = round(res7$results$x_est, 3),
        y_est = round(res7$results$y_est, 3),
        a_est = round(res7$results$a_est, 3),
        d_est = as.numeric(round(res7$results$d_est, 1)),
        fit_est = round(res7$results$fit_est, 3),
        labels = as.character(res7$results$Label),
        # CIs
        e_lci = round(res7$results$e_lci, 3),
        e_uci = round(res7$results$e_uci, 3),
        a_lci = round(res7$results$a_lci, 3),
        a_uci = round(res7$results$a_uci, 3)
    )
)
write_json(fixture7, file.path(output_dir, "ssm_group_contrast_correlation.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Test 8: SSM parameters function (no bootstrap)
# =============================================================================
cat("8. SSM parameters calculation...\n")

# Test data
test_scores <- c(0.374, -0.572, -0.520, 0.016, 0.688, 1.142, 1.578, 0.678)
test_angles <- octants()

params <- ssm_parameters(test_scores, test_angles)

fixture8 <- list(
    function_name = "ssm_parameters",
    input = list(
        scores = test_scores,
        angles = as.numeric(test_angles)
    ),
    expected = list(
        elevation = round(params[[1]], 6),
        x_value = round(params[[2]], 6),
        y_value = round(params[[3]], 6),
        amplitude = round(params[[4]], 6),
        displacement = round(params[[5]], 6), # radians
        fit = round(params[[6]], 6)
    )
)
write_json(fixture8, file.path(output_dir, "ssm_parameters.json"),
    auto_unbox = TRUE, pretty = TRUE
)

# =============================================================================
# Export example datasets as CSV
# =============================================================================
cat("\nExporting datasets to CSV...\n")
write.csv(jz2017, file.path(output_dir, "jz2017.csv"), row.names = FALSE)
write.csv(aw2009, file.path(output_dir, "aw2009.csv"), row.names = FALSE)

cat("\n✓ Test fixtures exported to", output_dir, "\n")
cat("✓ Datasets exported as CSV\n")
cat("\nFiles created:\n")
list.files(output_dir, full.names = FALSE)
