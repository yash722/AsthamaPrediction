library(caret)
library(e1071) 
library(Metrics)
library(randomForest)
library(xgboost)
library(rpart)
library(pROC)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(caTools)
library(caret)
library(Boruta)
library(kknn)
library(MASS)
library(rpart)
library(ada)

# Set Working Directory
setwd("C:/Users/yashr/OneDrive/Desktop/CollegeStuff/MET-CS-699-Data-Mining/Project/")
df <- read.csv("project_data.csv")
sapply(df, class)

# Calculate unique values for each column
unique_values <- df %>%
  summarise_all(~ n_distinct(.)) %>%
  gather(key = "column", value = "unique_values") %>%
  arrange(desc(unique_values))

print(unique_values)

# Removing all columns with no distinct values
df <- df %>%
  select_if(~ n_distinct(.) > 1)

# Removing DATE columns
df <- df %>%
  select(-c("IDATE", "IDATE_F"))

# Removing EMP_STAT, UNEMP_R, EMP_EVER1 (Status on Employment, not the work environment itself)
df <- df %>%
  select(-c("EMP_STAT", "UNEMP_R", "EMP_EVER1"))

# Removing SEQNO, SEQNO_FINL (Sequence Number removal)
df <- df %>%
  select(-c("SEQNO", "SEQNO_FINL"))

# Converting columns where distinct values are less than 33
df <- df %>%
  mutate(across(where(~ n_distinct(.) <= 33), as.factor))

# Class to 0 -> N, 1 -> Y
df$Class <- ifelse(df$Class == "Y", 1, 0)
df$Class <- as.factor(df$Class)

# Patient MetaData Analysis
# Histogram for AGEDX
ggplot(df, aes(x = AGEDX)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Age Distribution (AGEDX)", x = "Age", y = "Count")

# Bar Plot for SEXVAR
ggplot(df, aes(x = factor(SEXVAR))) +
  geom_bar(fill = "lightgreen", color = "black") +
  labs(title = "Gender Distribution (SEXVAR)", x = "Gender", y = "Count")

# Bar Plot for S_INSIDE
ggplot(df, aes(x = factor(S_INSIDE))) +
  geom_bar(fill = "orange", color = "black") +
  labs(title = "S_INSIDE Distribution", x = "S_INSIDE", y = "Count")

# Calculate percentage of NA values for each column
calculate_na_percentage <- function(df) {
  na_percentage <- df %>%
    summarise_all(~ mean(is.na(.)) * 100) %>%
    gather(key = "column", value = "percentage_NA") %>%
    arrange(desc(percentage_NA))
  
  return(na_percentage)
}

na_percentage <- calculate_na_percentage(df)
print(na_percentage)

# Function to calculate the percentage of specific values in columns
calculate_specific_value_percentage <- function(df, values) {
  result <- data.frame(Column = character(), `9` = numeric(), `99` = numeric(), `999` = numeric())
  factor_columns <- names(df)[sapply(df, is.factor)]
  # Handling Factor columns
  for (col in factor_columns) {
    percentages <- sapply(values, function(val) {
      count <- sum(df[[col]] == val, na.rm = TRUE)
      total <- nrow(df)
      (count / total) * 100
    })
    result <- rbind(result, data.frame(Column = col, `9` = percentages[1], `99` = percentages[2], `999` = percentages[3]))
  }
  
  # Handle numeric/integer columns
  numeric_columns <- names(df)[sapply(df, is.numeric)]
  for (col in numeric_columns) {
    percentages <- sapply(as.numeric(values), function(val) {
      count <- sum(df[[col]] == val, na.rm = TRUE)
      total <- nrow(df)
      (count / total) * 100
    })
    result <- rbind(result, data.frame(Column = col, `9` = percentages[1], `99` = percentages[2], `999` = percentages[3]))
  }
  
  return(result)
}

# Define the specific values you want to check -> Refused
values_to_check <- c("9", "99", "999")

result <- calculate_specific_value_percentage(df, values_to_check)
write.csv(result, file = "percentage_of_9_99_999.csv", row.names = FALSE)

# Display the result
print(result)

# NA handling
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Impute missing values in factor columns with the mode
factor_columns <- names(df)[sapply(df, is.factor)]
for (col in factor_columns) {
  mode_value <- get_mode(df[[col]])
  df[[col]][is.na(df[[col]])] <- mode_value
}

# Impute missing values in numeric/integer columns with the median
numeric_columns <- names(df)[sapply(df, is.numeric)]
for (col in numeric_columns) {
  median_value <- median(df[[col]], na.rm = TRUE)
  df[[col]][is.na(df[[col]])] <- median_value
}

na_percentage <- calculate_na_percentage(df)
print(na_percentage)

# Correlation Plot
# Select only numeric columns
numeric_columns <- df[, sapply(df, is.numeric)]

# Calculate the correlation matrix for numeric columns
cor_matrix <- cor(df[, sapply(df, is.numeric)], use = "complete.obs")

# Set the threshold for high correlations
threshold <- 0.7
high_cor_pairs <- which(cor_matrix > threshold | cor_matrix < -threshold, arr.ind = TRUE)
high_cor_pairs <- high_cor_pairs[high_cor_pairs[,1] != high_cor_pairs[,2], ]
high_cor_df <- data.frame(
  Column1 = rownames(cor_matrix)[high_cor_pairs[,1]],
  Column2 = colnames(cor_matrix)[high_cor_pairs[,2]],
  Correlation = cor_matrix[high_cor_pairs]
)
high_cor_df <- high_cor_df[!duplicated(t(apply(high_cor_df, 1, sort))), ]
print(high_cor_df)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Removing REPNUM, NRECSEL, HISPANC3, MARITAL, WEIGHT2, WTKG3, DISPCODE
# NRECSEL same as NRECSTR (collinearity)
# WEIGHT_IN same as LLCPWT_F (collinearity)
# HTM4 same as HTIN4 (collinearilty)
# DISPCODE same as DISPCODE_F (Same)
# HISPANC3 -> No basis
df <- df %>%
  select(-c("REPNUM", "NRECSEL", "WEIGHT_IN", "HTM4", "DISPCODE_F", "HISPANC3"))

# Outlier Detection
# Boxplots for Numeric variables

numeric_columns <- names(df)[sapply(df, is.numeric)]

for (col in numeric_columns) {
  # Create boxplots
  p <- ggplot(df, aes_string(x = "1", y = col)) + 
    geom_boxplot(fill = "lightgreen", color = "black") +
    labs(title = paste(col, "Before Capping"), x = "", y = col)
  
  # Display the boxplots
  print(p)
}

cap_outliers <- function(x) {
  lower_bound <- quantile(x, 0.01, na.rm = TRUE)
  upper_bound <- quantile(x, 0.99, na.rm = TRUE)
  
  # Cap outliers
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

# Apply capping to all numeric columns
for (col in numeric_columns) {
  # Apply outlier capping
  df[[col]] <- cap_outliers(df[[col]])
  
  # Create boxplots after capping
  p <- ggplot(df, aes_string(x = "1", y = col)) + 
    geom_boxplot(fill = "lightgreen", color = "black") +
    labs(title = paste(col, "After Capping"), x = "", y = col)
  
  # Display the boxplots
  print(p)
}

# After log transformations -> Higher number of outliers detected
no_log_transformations <- c("MISS_DAY", "POTATOE1", "FRENCHF1", "FVGREEN1")

df <- df %>%
  mutate(across(setdiff(numeric_columns, no_log_transformations), ~ log(. + 1)))

for (col in numeric_columns) {
  # Create boxplots after capping
  p <- ggplot(df, aes_string(x = "1", y = col)) + 
    geom_boxplot(fill = "lightgreen", color = "black") +
    labs(title = paste(col, "After Capping After Log Transformation"), x = "", y = col)
  
  # Display the boxplots
  print(p)
}

# Even after capping and log transformation -> no change in the status of outliers
df <- df %>%
  select(-c("WEIGHT2"))

numeric_columns <- names(df)[sapply(df, is.numeric)]

# Standardize numeric columns
df <- df %>%
  mutate(across(all_of(numeric_columns), ~ scale(.) %>% as.vector()))

# Function to detect rare categories in factor variables and show percentages in a table
detect_rare_categories <- function(x, threshold = 0.05) {
  freq_table <- table(x)
  total_count <- sum(freq_table)
  
  # Calculate the percentage of each category
  percentages <- freq_table / total_count * 100  # Convert to percentages
  
  # Create a data frame to store category names, percentages, and if they are rare
  data <- data.frame(
    Category = names(freq_table),
    Frequency = as.numeric(freq_table),
    Percentage = percentages,
    Is_Rare = percentages < (threshold * 100)
  )
  
  return(data)
}

# Specify the threshold (e.g., categories that make up less than 4.5% of the data)
threshold <- 0.045  # Adjust threshold as needed

# Identify factor columns
factor_columns <- names(df)[sapply(df, is.factor)]

# Detect rare categories and percentages in each factor variable and display as tables
rare_categories_table <- lapply(df[factor_columns], detect_rare_categories, threshold = threshold)

# Name the list elements with their corresponding column names
names(rare_categories_table) <- factor_columns

# Display the tables for each factor variable
for (col in names(rare_categories_table)) {
  cat("\nTable for Factor Variable:", col, "\n")
  print(rare_categories_table[[col]])
}

fac_hist <- tail(factor_columns, 5)
for (var in fac_hist) {
  p <- ggplot(df, aes_string(x = var)) +
    geom_bar(fill = "lightblue", color = "black") +
    labs(title = paste("Bar Plot of", var), x = var, y = "Count") +
    theme_minimal()
  
  # Print the plot
  print(p)
}

fac_hist <- head(factor_columns, 5)
for (var in fac_hist) {
  p <- ggplot(df, aes_string(x = var)) +
    geom_bar(fill = "lightblue", color = "black") +
    labs(title = paste("Bar Plot of", var), x = var, y = "Count") +
    theme_minimal()
  
  # Print the plot
  print(p)
}

df <- df %>% 
  select(-c("REGION_F"))

# Since these outliers represent the answers of the respondents and are valuable to the data
# We are not going to remove outliers for factor variables and will work with them.
# In case the factor variable represents a hindrance, we can use dummy variables and remove 
# Rare categories. Due to limited knowledge of the factor variables of the dataset as well as the domain 
# We feel that removing them would result in removal of important information

# Saving the the new CSV file
write.csv(df, file = "preprocessed_data.csv", row.names = FALSE)
data <- read.csv("preprocessed_data.csv")

set.seed(123)

# Split the data: 80% for training and 20% for testing
split <- sample.split(data$Class, SplitRatio = 0.8)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

# Check the dimensions of the train and test sets
print(dim(train))
print(dim(test))

write.csv(train, file = "initial_train.csv", row.names = FALSE)
write.csv(test, file = "initial_test.csv", row.names = FALSE)

train <- read.csv("initial_train.csv")

# Convert the target variable to a factor
train$Class <- as.factor(train$Class)

# ----------- Random Under-Sampling (RUS) -----------

# Calculate the number of samples in the minority class
minority_class_count <- min(table(train$Class))

# Perform Random Under-Sampling on the majority class
rus_indices <- unlist(lapply(levels(train$Class), function(class) {
  sample(which(train$Class == class), size = minority_class_count, replace = FALSE)
}))

# Create the undersampled training set
train_undersampled <- train[rus_indices, ]

# Check the class distribution after RUS
print("Class distribution after RUS:")
print(table(train_undersampled$Class))

# ---------- Boruta Feature Selection ----------
# Apply Boruta to select important features
boruta_model <- Boruta(Class ~ ., data = train_undersampled, doTrace = 2)
selected_features <- getSelectedAttributes(boruta_model, withTentative = TRUE)

# Print selected features
print("Selected features using Boruta:")
print(selected_features)

# Create a new training dataset with only the selected features
train_boruta <- train_undersampled[, c(selected_features, "Class")]

# Save the Boruta-selected datasets
write.csv(train_boruta, "rus_boruta_train.csv", row.names = FALSE)

# --- Feature Selection using Random Forest (via caret) ---

# Train a Random Forest model to assess variable importance
control <- trainControl(method = "none")
model <- train(Class ~ ., data = train_undersampled, method = "rf", trControl = control)

# Get feature importance
importance <- varImp(model, scale = FALSE)

# Print the feature importance
print("Feature Importance using Random Forest:")
print(importance)

# Select top features
top_features <- rownames(importance$importance)[order(-importance$importance$Overall)][1:20]

# Create a new train dataset with selected features
train_info_gain <- train_undersampled[, c(top_features, "Class")]

# Save the datasets to CSV files
write.csv(train_info_gain, "rus_info_gain_train.csv", row.names = FALSE)

# --- LDA for Feature Selection ---

# Identify and remove constant features
constant_features <- sapply(train_undersampled[, -which(names(train_undersampled) == "Class")], function(x) {
  length(unique(x)) == 1
})
train_undersampled <- train_undersampled[, !constant_features]

# Perform LDA on the filtered dataset
lda_model <- lda(Class ~ ., data = train_undersampled)

# Extract coefficients of linear discriminants
lda_coefficients <- lda_model$scaling

# Print all features selected and their coefficients
print("Features selected using LDA and their coefficients:")
print(lda_coefficients)

# Sort features by the absolute values of their coefficients in descending order
sorted_features <- rownames(lda_coefficients)[order(-abs(lda_coefficients[, 1]))]

# Select the top 20 features
top_lda_features <- sorted_features[1:20]

# Create a new train dataset with features selected by LDA
lda_features <- rownames(lda_coefficients)
train_lda <- train_undersampled[, c(top_lda_features, "Class")]

# Save the LDA-processed datasets to CSV files
write.csv(train_lda, "rus_lda_train.csv", row.names = FALSE)

# ------ Clustered Data Sampling ------ #
desired_majority_samples <- 600  # Adjust as needed
desired_minority_samples <- 600   # Adjust as needed

# Separate the majority and minority classes
majority_class <- train[train$Class == 0, ]
minority_class <- train[train$Class == 1, ]

# Perform k-means clustering on the majority class for undersampling
clusters <- kmeans(majority_class[, -ncol(majority_class)], centers = desired_majority_samples)
majority_class_sample <- majority_class[!duplicated(clusters$cluster), ]

set.seed(123)
minority_class_sample <- minority_class[sample(nrow(minority_class), desired_minority_samples), ]
train_undersampled <- rbind(majority_class_sample, minority_class_sample)

# ---------- Boruta Feature Selection ----------
# Apply Boruta to select important features
boruta_model <- Boruta(Class ~ ., data = train_undersampled, doTrace = 2)
selected_features <- getSelectedAttributes(boruta_model, withTentative = TRUE)

# Print selected features
print("Selected features using Boruta:")
print(selected_features)

# Create a new training dataset with only the selected features
train_boruta <- train_undersampled[, c(selected_features, "Class")]

# Save the Boruta-selected datasets
write.csv(train_boruta, "cluster_boruta_train.csv", row.names = FALSE)

# --- Feature Selection using Random Forest (via caret) ---

# Train a Random Forest model to assess variable importance
control <- trainControl(method = "none")
model <- train(Class ~ ., data = train_undersampled, method = "rf", trControl = control)

# Get feature importance
importance <- varImp(model, scale = FALSE)

# Print the feature importance
print("Feature Importance using Random Forest:")
print(importance)

# Select top 20 features
top_features <- rownames(importance$importance)[order(-importance$importance$Overall)][1:20]

# Create a new train dataset with selected features
train_info_gain <- train_undersampled[, c(top_features, "Class")]

# Save the datasets to CSV files
write.csv(train_info_gain, "cluster_info_gain.csv", row.names = FALSE)

# --- LDA for Feature Selection ---

# Load the MASS library for LDA
# Identify and remove constant features
constant_features <- sapply(train_undersampled[, -which(names(train_undersampled) == "Class")], function(x) {
  length(unique(x)) == 1
})
train_undersampled <- train_undersampled[, !constant_features]

# Perform LDA on the filtered dataset
lda_model <- lda(Class ~ ., data = train_undersampled)

# Extract coefficients of linear discriminants
lda_coefficients <- lda_model$scaling

# Sort features by the absolute values of their coefficients in descending order
sorted_features <- rownames(lda_coefficients)[order(-abs(lda_coefficients[, 1]))]

# Select the top 20 features
top_lda_features <- sorted_features[1:20]

print("Top 20 features selected using LDA and their coefficients:")
print(lda_coefficients[top_lda_features, , drop = FALSE])

train_lda <- train_undersampled[, c(top_lda_features, "Class")]

write.csv(train_lda, "cluster_lda_train.csv", row.names = FALSE)

calculate_mcc <- function(conf_matrix) {
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[1, 2])
  FN <- as.numeric(conf_matrix$table[2, 1])
  
  numerator <- (TP * TN - FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if (denominator == 0) {
    mcc <- 0
  } else {
    mcc <- numerator / denominator
  }
  return(mcc)
}

calculate_auc <- function(model, test_data, target_column) {
  test_data[[target_column]] <- factor(test_data[[target_column]], levels = c("Class0", "Class1"))
  
  prob_predictions <- predict(model, test_data, type = "prob")
  prob_class_1 <- prob_predictions[, "Class1"]
  
  label <- test_data[[target_column]]
  
  if (length(unique(label)) < 2) {
    stop("Both Class0 and Class1 should be present in the test set for ROC calculation.")
  }
  
  
  roc_obj <- roc(label, prob_class_1, levels = c("Class0", "Class1"), direction = "<")
  
  plot(roc_obj, main = "ROC Curve")
  auc_value <- auc(roc_obj)
  return(auc_value)
}

calculate_weighted_metrics <- function(conf_matrix_0, conf_matrix_1, auc, test_data, target_column) {
  class_counts <- table(test_data[[target_column]])
  weight_class_n <- class_counts["Class0"] / sum(class_counts)
  weight_class_y <- class_counts["Class1"] / sum(class_counts)
  
  tp_rate_class_n <- conf_matrix_0$byClass["Sensitivity"]
  fp_rate_class_n <- 1 - conf_matrix_0$byClass["Specificity"]
  precision_class_n <- conf_matrix_0$byClass["Pos Pred Value"]
  recall_class_n <- conf_matrix_0$byClass["Sensitivity"]
  f_measure_class_n <- conf_matrix_0$byClass["F1"]
  mcc_class_n <- calculate_mcc(conf_matrix_0)
  kappa_class_n <- conf_matrix_0$overall["Kappa"]
  
  # Extract metrics for Class Y (1)
  tp_rate_class_y <- conf_matrix_1$byClass["Sensitivity"]
  fp_rate_class_y <- 1 - conf_matrix_1$byClass["Specificity"]
  precision_class_y <- conf_matrix_1$byClass["Pos Pred Value"]
  recall_class_y <- conf_matrix_1$byClass["Sensitivity"]
  f_measure_class_y <- conf_matrix_1$byClass["F1"]
  mcc_class_y <- calculate_mcc(conf_matrix_1)
  kappa_class_y <- conf_matrix_1$overall["Kappa"]
  
  # Calculate weighted averages
  weighted_tp_rate <- round(tp_rate_class_n * weight_class_n + tp_rate_class_y * weight_class_y, 3)
  weighted_fp_rate <- round(fp_rate_class_n * weight_class_n + fp_rate_class_y * weight_class_y, 3)
  weighted_precision <- round(precision_class_n * weight_class_n + precision_class_y * weight_class_y, 3)
  weighted_recall <- round(recall_class_n * weight_class_n + recall_class_y * weight_class_y, 3)
  weighted_f_measure <- round(f_measure_class_n * weight_class_n + f_measure_class_y * weight_class_y, 3)
  weighted_mcc <- round(mcc_class_n * weight_class_n + mcc_class_y * weight_class_y, 3)
  weighted_kappa <- round(kappa_class_n * weight_class_n + kappa_class_y * weight_class_y, 3)
  
  # Print weighted metrics
  cat("\nWeighted Metrics:\n")
  cat("Weighted TP Rate (Recall):", weighted_tp_rate, "\n")
  cat("Weighted FP Rate:", weighted_fp_rate, "\n")
  cat("Weighted Precision:", weighted_precision, "\n")
  cat("Weighted Recall:", weighted_recall, "\n")
  cat("Weighted F-measure:", weighted_f_measure, "\n")
  cat("Weighted MCC:", weighted_mcc, "\n")
  cat("Weighted Kappa Statistic:", weighted_kappa, "\n")
  cat("Weighted AUC-ROC:", auc, "\n")
}


evaluate_model <- function(model, test_data, target_column) {
  roc_area <- round(calculate_auc(model, test_data, target_column), 3)
  test_data[[target_column]] <- factor(test_data[[target_column]], levels = c("Class0", "Class1"))
  predictions <- predict(model, test_data)
  
  conf_matrix_0 <- confusionMatrix(predictions, test_data[[target_column]], positive = "Class0")
  conf_matrix_1 <- confusionMatrix(predictions, test_data[[target_column]], positive = "Class1")
  
  # Extract metrics for Class N (0)
  tp_rate_class_n <- round(conf_matrix_0$byClass["Sensitivity"], 3)  # True Positive Rate (Recall)
  fp_rate_class_n <- round(1 - conf_matrix_0$byClass["Specificity"], 3)  # False Positive Rate
  precision_class_n <- round(conf_matrix_0$byClass["Pos Pred Value"], 3)  # Precision
  recall_class_n <- round(conf_matrix_0$byClass["Sensitivity"], 3)  # Recall
  f_measure_class_n <- round(conf_matrix_0$byClass["F1"], 3) # F1 Score
  mcc_class_n <- round(calculate_mcc(conf_matrix_0), 3)  # Matthews Correlation Coefficient
  kappa_class_n <- round(conf_matrix_0$overall["Kappa"], 3)  # Kappa Statistic
  
  # Extract metrics for Class Y (1)
  tp_rate_class_y <- round(conf_matrix_1$byClass["Sensitivity"], 3)  # True Positive Rate (Recall)
  fp_rate_class_y <- round(1 - conf_matrix_1$byClass["Specificity"], 3)  # False Positive Rate
  precision_class_y <- round(conf_matrix_1$byClass["Pos Pred Value"], 3)  # Precision
  recall_class_y <- round(conf_matrix_1$byClass["Sensitivity"], 3)  # Recall
  f_measure_class_y <- round(conf_matrix_1$byClass["F1"], 3)  # F1 Score
  mcc_class_y <- round(calculate_mcc(conf_matrix_1), 3)  # Matthews Correlation Coefficient
  kappa_class_y <- round(conf_matrix_1$overall["Kappa"], 3)  # Kappa Statistic
  
  # Print metrics for Class N (0)
  cat("Metrics for Class N (0):\n")
  cat("TP Rate (Recall):", tp_rate_class_n, "\n")
  cat("FP Rate:", fp_rate_class_n, "\n")
  cat("Precision:", precision_class_n, "\n")
  cat("Recall:", recall_class_n, "\n")
  cat("F-measure:", f_measure_class_n, "\n")
  cat("MCC:", mcc_class_n, "\n")
  cat("Kappa Statistic:", kappa_class_n, "\n")
  cat("AUC-ROC for Class 0:", roc_area, "\n\n")
  
  # Print metrics for Class Y (1)
  cat("Metrics for Class Y (1):\n")
  cat("TP Rate (Recall):", tp_rate_class_y, "\n")
  cat("FP Rate:", fp_rate_class_y, "\n")
  cat("Precision:", precision_class_y, "\n")
  cat("Recall:", recall_class_y, "\n")
  cat("F-measure:", f_measure_class_y, "\n")
  cat("MCC:", mcc_class_y, "\n")
  cat("Kappa Statistic:", kappa_class_y, "\n")
  cat("AUC-ROC for Class 1:", roc_area, "\n\n")
  
  calculate_weighted_metrics(conf_matrix_0, conf_matrix_1, roc_area, test_data, target_column)
}

rus_b <- read.csv("rus_boruta_train.csv")
rus_lda <- read.csv("rus_lda_train.csv")
rus_info_gain <- read.csv("rus_info_gain_train.csv")

cluster_b <- read.csv("cluster_boruta_train.csv")
cluster_lda <- read.csv("cluster_lda_train.csv")
cluster_info_gain <- read.csv("cluster_info_gain.csv")

test_data <- read.csv("initial_test.csv")
target_var <- "Class"

rus_b[[target_var]] <- as.factor(rus_b[[target_var]])
rus_info_gain[[target_var]] <- as.factor(rus_info_gain[[target_var]])
rus_lda[[target_var]] <- as.factor(rus_lda[[target_var]])

cluster_b[[target_var]] <- as.factor(cluster_b[[target_var]])
cluster_lda[[target_var]] <- as.factor(cluster_lda[[target_var]])
cluster_info_gain[[target_var]] <- as.factor(cluster_info_gain[[target_var]])

test_data[[target_var]] <- as.factor(test_data[[target_var]])


rus_b[[target_var]] <- factor(rus_b[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))
rus_info_gain[[target_var]] <- factor(rus_info_gain[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))
rus_lda[[target_var]] <- factor(rus_lda[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))

cluster_b[[target_var]] <- factor(cluster_b[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))
cluster_info_gain[[target_var]] <- factor(cluster_info_gain[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))
cluster_lda[[target_var]] <- factor(cluster_lda[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))

test_data[[target_var]] <- factor(test_data[[target_var]], levels = c("0", "1"), labels = c("Class0", "Class1"))

print(table(rus_info_gain[[target_var]]))
print(table(cluster_info_gain[[target_var]]))

## Classification and Testing ##
## KNN ##
train_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  search = "grid",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# grid for hyperparameter tuning
tune_grid_knn <- expand.grid(kmax = seq(1, 20, by = 2),  # range of k values
                             distance = 1,              # 1 for Manhattan distance
                             kernel = "rectangular")
## RUS ##
# LDA
knn_rus_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_lda,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_rus_lda$bestTune)
print(knn_rus_lda)

evaluate_model(knn_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
knn_rus_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_info_gain,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_rus_ig$bestTune)
print(knn_rus_ig)

evaluate_model(knn_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
knn_rus_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_b,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_rus_b$bestTune)
print(knn_rus_b)

evaluate_model(knn_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
knn_c_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_lda,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_c_lda$bestTune)
print(knn_c_lda)

evaluate_model(knn_c_lda, test_data, target_var)

# Info Gain
set.seed(42)
knn_c_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_info_gain,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_c_ig$bestTune)
print(knn_c_ig)

evaluate_model(knn_c_ig, test_data, target_var)

# Boruta
set.seed(42)
knn_c_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_b,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_c_b$bestTune)
print(knn_c_b)

evaluate_model(knn_c_b, test_data, target_var)

## Decision Tree (Rpart) ##
train_control <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  search = "grid", 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

tune_grid_rpart <- expand.grid(cp = seq(0.001, 0.05, by = 0.01)) 

## RUS ##
# LDA
rpart_rus_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_lda,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_rus_lda$bestTune)
print(rpart_rus_lda)

evaluate_model(rpart_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
rpart_rus_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_info_gain,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_rus_ig$bestTune)
print(rpart_rus_ig)

evaluate_model(rpart_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
rpart_rus_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_b,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_rus_b$bestTune)
print(rpart_rus_b)

evaluate_model(rpart_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
rpart_c_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_lda,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_c_lda$bestTune)
print(rpart_c_lda)

evaluate_model(rpart_c_lda, test_data, target_var)

# Info Gain
set.seed(42)
rpart_c_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_info_gain,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_c_ig$bestTune)
print(rpart_c_ig)

evaluate_model(rpart_c_ig, test_data, target_var)

# Boruta
set.seed(42)
rpart_c_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_b,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_rpart,
  metric = "ROC"
)

print(rpart_c_b$bestTune)
print(rpart_c_b)

evaluate_model(rpart_c_b, test_data, target_var)

## Adaboost ##
train_control <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  search = "grid", 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

tune_grid_ada <- expand.grid(
  iter = c(20, 50, 100),    
  maxdepth = c(3, 5, 7),  
  nu = c(0.1, 0.3, 0.5) 
)

## RUS ##
# LDA
ada_rus_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_lda,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_rus_lda$bestTune)
print(ada_rus_lda)

evaluate_model(ada_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
ada_rus_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_info_gain,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_rus_ig$bestTune)
print(ada_rus_ig)

evaluate_model(ada_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
ada_rus_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_b,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_rus_b$bestTune)
print(ada_rus_b)

evaluate_model(ada_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
ada_c_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_lda,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_c_lda$bestTune)
print(ada_c_lda)

evaluate_model(ada_c_lda, test_data, target_var)

# Info Gain
set.seed(42)
ada_c_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_info_gain,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_c_ig$bestTune)
print(ada_c_ig)

evaluate_model(ada_c_ig, test_data, target_var)

# Boruta
set.seed(42)
ada_c_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_b,
  method = "ada",
  trControl = train_control,
  tuneGrid = tune_grid_ada,
  metric = "ROC"
)

print(ada_c_b$bestTune)
print(ada_c_b)

evaluate_model(ada_c_b, test_data, target_var)

## Random Forest ##
train_control <- trainControl(method = "repeatedcv", 
                              number = 10,
                              repeats = 5,
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary)
rf_grid <- expand.grid(
  mtry = c(2:11)
)
## RUS ##
# LDA
set.seed(42)
rf_model_rus_lda <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_lda, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_rus_lda$bestTune)
print(rf_model_rus_lda)

evaluate_model(rf_model_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
rf_model_rus_ig <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_info_gain, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_rus_ig$bestTune)
print(rf_model_rus_ig)

evaluate_model(rf_model_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
rf_model_rus_b <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_b, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_rus_b$bestTune)
print(rf_model_rus_b)

evaluate_model(rf_model_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
rf_model_cls_lda <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_lda, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_cls_lda$bestTune)
print(rf_model_cls_lda)

evaluate_model(rf_model_cls_lda, test_data, target_var)

# Info Gain
set.seed(42)
rf_model_cls_ig <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_info_gain, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_cls_ig$bestTune)
print(rf_model_cls_ig)

evaluate_model(rf_model_cls_ig, test_data, target_var)

# Boruta
set.seed(42)
rf_model_cls_b <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_b, 
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  classwt = c(0.9, 0.1),
  ntree = 500
)

print(rf_model_cls_b$bestTune)
print(rf_model_cls_b)

evaluate_model(rf_model_cls_b, test_data, target_var)

## SVM RBF ##
train_control <- trainControl(
  method = "repeatedcv",  
  number = 5,  
  repeats = 3,
  summaryFunction = twoClassSummary, 
  classProbs = TRUE
)

svm_grid <- expand.grid(
  C = 2^(-2:2),                   
  sigma = 2^(-2:2)
)

## RUS ##
# LDA
set.seed(42)
svm_rus_lda <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_lda, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_rus_lda$bestTune)
print(svm_rus_lda)

evaluate_model(svm_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
svm_rus_ig <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_info_gain, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_rus_ig$bestTune)
print(svm_rus_ig)

evaluate_model(svm_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
svm_rus_b <- train(
  as.formula(paste(target_var, "~ .")), 
  data = rus_b, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_rus_b$bestTune)
print(svm_rus_b)

evaluate_model(svm_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
svm_c_lda <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_lda, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_c_lda$bestTune)
print(svm_c_lda)

evaluate_model(svm_c_lda, test_data, target_var)

# Info Gain
set.seed(42)
svm_c_ig <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_info_gain, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_c_ig$bestTune)
print(svm_c_ig)

evaluate_model(svm_c_ig, test_data, target_var)

# Boruta
set.seed(42)
svm_c_b <- train(
  as.formula(paste(target_var, "~ .")), 
  data = cluster_b, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

print(svm_c_b$bestTune)
print(svm_c_b)

evaluate_model(svm_c_b, test_data, target_var)

## XGBoost ##
train_control <- trainControl(
  method = "repeatedcv",  
  number = 5,  
  repeats = 3,
  summaryFunction = twoClassSummary, 
  classProbs = TRUE
)

xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1),
  colsample_bytree = c(0.5, 0.7),
  min_child_weight = c(1, 5),
  subsample = c(0.7, 1)
)

## RUS ##
# LDA
set.seed(42)
xgb_rus_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_lda, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_rus_lda$bestTune)
print(xgb_rus_lda)

evaluate_model(xgb_rus_lda, test_data, target_var)

# Info Gain
set.seed(42)
xgb_rus_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_info_gain, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_rus_ig$bestTune)
print(xgb_rus_ig)

evaluate_model(xgb_rus_ig, test_data, target_var)

# Boruta
set.seed(42)
xgb_rus_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = rus_b, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_rus_b$bestTune)
print(xgb_rus_b)

evaluate_model(xgb_rus_b, test_data, target_var)

## Clustered ##
# LDA
set.seed(42)
xgb_c_lda <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_lda, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_c_lda$bestTune)
print(xgb_c_lda)

evaluate_model(xgb_c_lda, test_data, target_var)

# Info Gain
set.seed(42)
xgb_c_ig <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_info_gain, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_c_ig$bestTune)
print(xgb_c_ig)

evaluate_model(xgb_c_ig, test_data, target_var)

# Boruta
set.seed(42)
xgb_c_b <- train(
  as.formula(paste(target_var, "~ .")),
  data = cluster_b, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_grid, 
  metric = "ROC"
)
print(xgb_c_b$bestTune)
print(xgb_c_b)

evaluate_model(xgb_c_b, test_data, target_var)
