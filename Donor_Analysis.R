############################################################
# Colorado Mesa University Foundation Donor Analytics
# Predicting Recurring Donor Behavior
# Author: [Your Name]
# Date: [Current Date]
############################################################

#-----------------------------------------------------------
# 1. Install and Load Packages
#-----------------------------------------------------------
required_packages <- c(
  "tidyverse", "readxl", "skimr", "psych", "caret", "randomForest",
  "pROC", "ggplot2", "maps", "mapdata", "knitr", "kableExtra",
  "broom", "corrplot", "cluster", "MASS", "fitdistrplus"
)

installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}

lapply(required_packages, library, character.only = TRUE)
set.seed(123)

#-----------------------------------------------------------
# 2. Load Dataset
#-----------------------------------------------------------
donor <- read.csv("DonorData.csv")

# Quick data overview
glimpse(donor)
skim(donor)

#-----------------------------------------------------------
# 3. Summarize Key Variables
#-----------------------------------------------------------
key_vars <- c(
  "Recognition.Gift.Total",
  "Recognition.Gift.Total.This.Year",
  "Recognition.Gift.Total.Last.Ten.Years",
  "Current.Consecutive.Giving.Streak"
)

psych::describe(donor[, key_vars])

#-----------------------------------------------------------
# 4. Create RecurringDonor Binary Variable
#-----------------------------------------------------------
donor <- donor %>%
  mutate(RecurringDonor = if_else(Current.Consecutive.Giving.Streak >= 2, 1, 0))

#-----------------------------------------------------------
# 5. Summary Statistics Table
#-----------------------------------------------------------
num_cols <- sapply(donor, is.numeric)

num_stats <- data.frame(
  Variable = names(donor)[num_cols],
  Mean = sapply(donor[, num_cols], function(x) mean(x, na.rm = TRUE)),
  Median = sapply(donor[, num_cols], function(x) median(x, na.rm = TRUE)),
  SD = sapply(donor[, num_cols], function(x) sd(x, na.rm = TRUE)),
  Min = sapply(donor[, num_cols], function(x) min(x, na.rm = TRUE)),
  Max = sapply(donor[, num_cols], function(x) max(x, na.rm = TRUE))
)

kable(num_stats, digits = 2, caption = "Summary Statistics for Numeric Variables") %>%
  kable_styling(full_width = FALSE)

#-----------------------------------------------------------
# 6. Visualizations
#-----------------------------------------------------------
ggplot(donor, aes(x = Recognition.Gift.Total.This.Year)) +
  geom_histogram(binwidth = 50, fill = "darkgreen", color = "white") +
  labs(title = "Distribution of Gift Totals This Year",
       x = "Gift Amount (This Year)", y = "Count") +
  theme_minimal()
ggsave("hist_gift_this_year.png", width = 8, height = 5, dpi = 300)

if("Constituent.Codes" %in% names(donor)){
  ggplot(donor, aes(x = as.factor(Constituent.Codes), y = Recognition.Gift.Total.This.Year)) +
    geom_boxplot(fill = "lightblue") +
    labs(title = "Gift Totals by Donor Type", x = "Donor Type", y = "Gift Total") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave("boxplot_gift_by_donortype.png", width = 8, height = 5, dpi = 300)
}

#-----------------------------------------------------------
# 7. Correlation Matrix
#-----------------------------------------------------------
num_df <- donor[, num_cols]
cor_matrix <- cor(num_df, use = "pairwise.complete.obs")
corrplot::corrplot(cor_matrix, method = "color", type = "upper")

#-----------------------------------------------------------
# 8. Prepare Data for Modeling
#-----------------------------------------------------------
df <- donor 
target <- "RecurringDonor"
predictors <- c("Recognition.Gift.Total", 
                "Current.Consecutive.Giving.Streak", 
                "Recognition.Gift.Total.This.Year", 
                "Recognition.Gift.Total.Last.Year", 
                "Recognition.Gift.Total.Two.Years.Ago", 
                "Recognition.Gift.Total.Three.Years.Ago", 
                "Recognition.Gift.Total.Four.Years.Ago", 
                "Recognition.Gift.Total.Five.Years.Ago", 
                "Recognition.Gift.Total.Six.Years.Ago", 
                "Recognition.Gift.Total.Seven.Years.Ago", 
                "Recognition.Gift.Total.Eight.Years.Ago", 
                "Recognition.Gift.Total.Nine.Years.Ago", 
                "Recognition.Gift.Total.Last.Ten.Years") 

# Convert dependent variable to binary numeric 
if (is.character(df[[target]]) || is.factor(df[[target]])) { df[[target]] <- ifelse(df[[target]] %in% c("Yes", "Y", "1", "Recurring", "True"), 1, 0) } else { df[[target]] <- as.numeric(df[[target]]) }

# Keep only complete cases 
model_data <- df[, c(target, predictors)]
model_data <- na.omit(model_data)

# Split into training and test sets 
set.seed(123) 
train_index <- createDataPartition(model_data[[target]], p = 0.8, list = FALSE) 
train <- model_data[train_index, ] 
test <- model_data[-train_index, ] 


#-----------------------------------------------------------
# 9. Logistic Regression Model
#-----------------------------------------------------------

# Fit logistic regression model 
logit_formula <- as.formula(paste(target, "~", paste(predictors, collapse = " + "))) 
logit_model <- glm(logit_formula, data = train, family = binomial(link = "logit")) 

# Model summary 

summary(logit_model)


# Predict probabilities and classes
test$Predicted_Prob <- predict(logit_model, newdata = test, type = "response")

# Match factor levels
test$Predicted_Class <- ifelse(test$Predicted_Prob > 0.5, 1, 0)

# Confusion matrix
test$Predicted_Class <- factor(test$Predicted_Class, levels = c(0, 1))
test[[target]] <- factor(test[[target]], levels = c(0, 1))

conf_matrix <- confusionMatrix(test$Predicted_Class, test[[target]])
print(conf_matrix)
# ROC curve
roc_logit <- roc(response = as.numeric(test$Predicted_Class, levels = c(0, 1)),
                 predictor = test$Predicted_Prob)
plot(roc_logit, col = "blue", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc(roc_logit), 3), ")"))
ggsave("roc_logistic.png", width = 7, height = 5, dpi = 300)

# Coefficients and odds ratios
coef_table <- broom::tidy(logit_model)
coef_table$Odds_Ratio <- exp(coef_table$estimate)
coef_table <- coef_table[, c("term", "estimate", "Odds_Ratio", "p.value")]

write.csv(coef_table, "logistic_coefficients.csv", row.names = FALSE)

#-----------------------------------------------------------
# 10. Random Forest Model
#-----------------------------------------------------------
rf_model <- randomForest(as.factor(RecurringDonor) ~ ., data = train, ntree = 500, importance = TRUE)
print(rf_model)
varImpPlot(rf_model, main = "Variable Importance - Random Forest")
ggsave("rf_varimp.png", width = 7, height = 5, dpi = 300)

rf_pred <- predict(rf_model, newdata = test, type = "prob")[,2]
rf_class <- predict(rf_model, newdata = test, type = "response")
confusionMatrix(rf_class, test$RecurringDonor)

roc_rf <- roc(as.numeric(as.character(test$RecurringDonor)), rf_pred)
plot(roc_rf, main = paste("Random Forest ROC (AUC =", round(auc(roc_rf),3), ")"))
ggsave("roc_rf.png", width = 7, height = 5, dpi = 300)

#-----------------------------------------------------------
# 11. ZIP Code Regression
#-----------------------------------------------------------
zip_summary <- donor %>%
  group_by(Mailing.Zip.Postal.Code) %>%
  summarise(AverageGift = mean(Recognition.Gift.Total, na.rm = TRUE),
            RecurringRate = mean(RecurringDonor, na.rm = TRUE),
            DonorCount = n())

zip_lm <- lm(RecurringRate ~ AverageGift, data = zip_summary)
summary(zip_lm)

ggplot(zip_summary, aes(x = AverageGift, y = RecurringRate)) +
  geom_point(color = "darkgreen") +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Recurring Donor Rate by ZIP Code",
       x = "Average Gift by ZIP Code", y = "Recurring Donor Rate")
ggsave("zip_regression.png", width = 8, height = 5, dpi = 300)

#-----------------------------------------------------------
# 12. Donor Segmentation (K-Means Clustering)
#-----------------------------------------------------------
cluster_data <- donor[, c("Recognition.Gift.Total.Last.Ten.Years",
                          "Current.Consecutive.Giving.Streak",
                          "Recognition.Gift.Total.This.Year")]
cluster_data <- scale(cluster_data)

kmeans_result <- kmeans(cluster_data, centers = 3, nstart = 25)
donor$Cluster <- as.factor(kmeans_result$cluster)

ggplot(donor, aes(x = Recognition.Gift.Total.Last.Ten.Years, 
                  y = Current.Consecutive.Giving.Streak, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Donor Segmentation by Giving Behavior",
       x = "Total Gift (10 Years)", y = "Consecutive Giving Streak")
ggsave("kmeans_clusters.png", width = 8, height = 5, dpi = 300)

#-----------------------------------------------------------
# 13. Save Models and Tables
#-----------------------------------------------------------
saveRDS(logit_model, "logit_model.rds")
saveRDS(rf_model, "rf_model.rds")
