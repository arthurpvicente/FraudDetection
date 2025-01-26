library(tidyverse)
library(caret)
library(cluster)
library(factoextra)

# Load the dataset
data <- read.csv("bank_transactions_data_2.csv")

cat("Summary of the dataset:\n")
summary(data)

# Check for missing values
cat("\nMissing values:\n")
print(colSums(is.na(data)))

# Histograms to view the distribution of key variables
ggplot(data, aes(x = TransactionAmount)) + 
  geom_histogram(bins = 30, fill = "blue", color = "black") + 
  labs(title = "Distribution of Transaction Amount")

ggplot(data, aes(x = AccountBalance)) + 
  geom_histogram(bins = 30, fill = "green", color = "black") + 
  labs(title = "Distribution of Account Balance")

# Scatterplot to explore relationships between variables
ggplot(data, aes(x = TransactionAmount, y = AccountBalance)) +
  geom_point(alpha = 0.7) +
  labs(title = "Scatterplot: Transaction Amount vs Account Balance")

# Outlier detection using IQR method for TransactionAmount
Q1 <- quantile(data$TransactionAmount, 0.25)
Q3 <- quantile(data$TransactionAmount, 0.75)
IQR <- Q3 - Q1
outliers <- data |> filter(TransactionAmount < (Q1 - 1.5 * IQR) | TransactionAmount > (Q3 + 1.5 * IQR))
cat("\nOutliers detected for TransactionAmount:\n")
print(outliers)

# Normalize the dataset
numeric_cols <- sapply(data, is.numeric)
numeric_data <- data |> select(where(is.numeric))
numeric_scaled <- scale(numeric_data)

# Handle missing values
numeric_cols <- sapply(data, is.numeric)
categorical_cols <- sapply(data, is.character)

# Numeric columns with median
data[numeric_cols] <- lapply(data[numeric_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Fill categorical columns with mode
for (col in names(data)[categorical_cols]) {
  mode_value <- as.character(sort(table(data[[col]]), decreasing = TRUE)[1])
  data[[col]] <- ifelse(is.na(data[[col]]), mode_value, data[[col]])
}

# Objective: Detect fraudulent transaction patterns using clustering
numeric_scaled <- scale(data |> select(where(is.numeric)))

# Perform K-Means clustering
set.seed(42)
kmeans_result <- kmeans(numeric_scaled, centers = 5, nstart = 25)
data$KMeans_Cluster <- kmeans_result$cluster

# Calculate distances from centroids.
# Centroids refer to the central points of clusters in a clustering algorithm.
centroids <- kmeans_result$centers
distances <- apply(numeric_scaled, 1, function(x) min(sqrt(colSums((t(centroids) - x)^2))))
data$KMeans_Distance <- distances

# Mean + 3 * Standard Deviation: Identifying Fraud
# The threshold for identifying potential fraud is set by calculating the mean distance from the centroids (the center of the clusters)
# and adding 3 times the standard deviation of the distances. This threshold is commonly used in anomaly detection, as points that
# fall further than 3 standard deviations from the mean are considered outliers (fraudulent transactions, in this case).
threshold <- mean(distances) + 3 * sd(distances)

# Flagging Fraudulent Transactions:
# The distances variable contains the distance of each point (transaction) from its cluster centroid.
# Any transaction with a distance greater than the threshold is flagged as potentially fraudulent.
# The result is stored in the new column KMeans_Fraud, where TRUE indicates a fraud, and FALSE indicates a normal transaction.
data$KMeans_Fraud <- distances > threshold

# Summary
fraud_summary <- table(data$KMeans_Fraud)
cat("\nFraud Summary:\n")
print(fraud_summary)

# The objective of this part of the code is to visualize how the data points have been grouped into clusters based on 
# the features TransactionAmount and AccountBalance, which were used in the 
#K-Means clustering algorithm. This helps in understanding how well the clustering algorithm has separated the data.
# Visualize K-Means Clusters with two key features: TransactionAmount and AccountBalance
ggplot(data, aes(x = TransactionAmount, y = AccountBalance, color = as.factor(KMeans_Cluster))) +
  geom_point(alpha = 0.7) +
  labs(title = "K-Means clustering algorithm", x = "Transaction Amount", y = "Account Balance") +
  scale_color_viridis_d() +
  theme_minimal()

# Highlight fraud points
# Red points are fraud detected
fraud_points <- data |> filter(KMeans_Fraud == TRUE)
ggplot(data, aes(x = TransactionAmount, y = AccountBalance, color = as.factor(KMeans_Cluster))) +
  geom_point(alpha = 0.7) +
  geom_point(data = fraud_points, aes(x = TransactionAmount, y = AccountBalance), color = "red", size = 3) +
  labs(title = "K-Means Clusters with Fraud Points", x = "Transaction Amount", y = "Account Balance") +
  scale_color_viridis_d() +
  theme_minimal()

# Fraud detection logic is based on distance from centroids
cat("\nTotal Fraudulent Transactions Detected (Using K-Means clustering):", nrow(fraud_points), "\n")
cat("\nFraudulent Transactions Detected:\n")
print(fraud_points)


# 'fraud_summary' is the variable containing the result of the fraud detection.
cat("Fraud Summary:\n", fraud_summary)
