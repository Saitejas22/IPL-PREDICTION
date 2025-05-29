# IPL Match Winner Prediction - R Shiny Application
# Load required libraries
library(shiny)
library(shinythemes)
library(dplyr)
library(tidyr)
library(readr)
library(caret)
library(randomForest)

# Define current IPL teams
CURRENT_TEAMS <- c(
  'Chennai Super Kings',
  'Delhi Capitals',
  'Gujarat Titans',
  'Kolkata Knight Riders',
  'Lucknow Super Giants',
  'Mumbai Indians',
  'Punjab Kings',
  'Rajasthan Royals',
  'Royal Challengers Bengaluru',
  'Sunrisers Hyderabad'
)

# Define venues
VENUES <- c(
  'M Chinnaswamy Stadium',
  'Wankhede Stadium',
  'MA Chidambaram Stadium',
  'Arun Jaitley Stadium',
  'Narendra Modi Stadium',
  'Eden Gardens',
  'Ekana Cricket Stadium',
  'Punjab Cricket Association Stadium',
  'Sawai Mansingh Stadium',
  'Rajiv Gandhi International Stadium'
)

# Map venues to cities
venue_city_map <- list(
  'M Chinnaswamy Stadium' = 'Bangalore',
  'Wankhede Stadium' = 'Mumbai',
  'MA Chidambaram Stadium' = 'Chennai',
  'Arun Jaitley Stadium' = 'Delhi',
  'Narendra Modi Stadium' = 'Ahmedabad',
  'Eden Gardens' = 'Kolkata',
  'Ekana Cricket Stadium' = 'Lucknow',
  'Punjab Cricket Association Stadium' = 'Mohali',
  'Sawai Mansingh Stadium' = 'Jaipur',
  'Rajiv Gandhi International Stadium' = 'Hyderabad'
)

# Function to load data and train model
train_model <- function() {
  # Check if data file exists, if not create sample data
  if (file.exists("ipl_2025.csv")) {
    ipl_data <- read_csv("ipl_2025.csv")
  } else {
    # Load original data
    if (file.exists("ipl.csv")) {
      ipl_data <- read_csv("ipl.csv")
      
      # Update team names
      team_mappings <- c(
        'Royal Challengers Bangalore' = 'Royal Challengers Bengaluru',
        'Kings XI Punjab' = 'Punjab Kings',
        'Delhi Daredevils' = 'Delhi Capitals'
      )
      
      for (old_name in names(team_mappings)) {
        ipl_data$team1[ipl_data$team1 == old_name] <- team_mappings[old_name]
        ipl_data$team2[ipl_data$team2 == old_name] <- team_mappings[old_name]
        ipl_data$winner[ipl_data$winner == old_name] <- team_mappings[old_name]
        ipl_data$toss_winner[ipl_data$toss_winner == old_name] <- team_mappings[old_name]
      }
      
      # Filter to current teams only
      ipl_data <- ipl_data %>%
        filter(team1 %in% CURRENT_TEAMS & team2 %in% CURRENT_TEAMS)
      
      # Save filtered data
      write_csv(ipl_data, "ipl_2025.csv")
    } else {
      # Create sample data
      set.seed(123)
      sample_size <- 100
      
      ipl_data <- data.frame(
        team1 = sample(CURRENT_TEAMS, sample_size, replace = TRUE),
        team2 = sample(CURRENT_TEAMS, sample_size, replace = TRUE),
        venue = sample(VENUES, sample_size, replace = TRUE),
        city = NA,
        toss_winner = NA,
        toss_decision = sample(c("bat", "field"), sample_size, replace = TRUE),
        winner = NA,
        stringsAsFactors = FALSE
      )
      
      # Fill in city based on venue
      for (i in 1:nrow(ipl_data)) {
        venue <- ipl_data$venue[i]
        ipl_data$city[i] <- venue_city_map[[venue]]
      }
      
      # Ensure team1 != team2
      for (i in 1:nrow(ipl_data)) {
        while (ipl_data$team1[i] == ipl_data$team2[i]) {
          ipl_data$team2[i] <- sample(CURRENT_TEAMS, 1)
        }
      }
      
      # Assign toss winners and match winners
      for (i in 1:nrow(ipl_data)) {
        teams <- c(ipl_data$team1[i], ipl_data$team2[i])
        ipl_data$toss_winner[i] <- sample(teams, 1)
        ipl_data$winner[i] <- sample(teams, 1, prob = c(0.55, 0.45))  # Slight advantage to team1
      }
      
      write_csv(ipl_data, "ipl_2025.csv")
    }
  }
  
  # Create a complete dataset with all possible team combinations
  set.seed(123)
  all_combinations <- expand.grid(
    team1 = CURRENT_TEAMS,
    team2 = CURRENT_TEAMS,
    venue = VENUES,
    toss_winner = c("team1", "team2"),
    toss_decision = c("bat", "field"),
    stringsAsFactors = FALSE
  ) %>%
  filter(team1 != team2) %>%  # Remove matches where team1 == team2
  mutate(
    # Convert placeholder values to actual team names
    toss_winner = ifelse(toss_winner == "team1", team1, team2)
  )
  
  # Add dummy outcomes (these won't be used for training, just to ensure all factor levels exist)
  dummy_data <- all_combinations %>%
    mutate(
      winner = ifelse(runif(n()) > 0.5, team1, team2),
      city = sapply(venue, function(v) venue_city_map[[v]])
    )
  
  # Combine with actual data
  # We'll use a very small weight for dummy data so it doesn't affect the model much
  combined_data <- bind_rows(
    # Actual data with weight 0.999
    ipl_data %>% mutate(weight = 0.999),
    # Dummy data with weight 0.001
    dummy_data %>% mutate(weight = 0.001)
  )
  
  # Prepare data for model training
  model_data <- combined_data %>%
    mutate(
      team1_factor = factor(team1, levels = CURRENT_TEAMS),
      team2_factor = factor(team2, levels = CURRENT_TEAMS),
      venue_factor = factor(venue, levels = VENUES),
      toss_winner_factor = factor(toss_winner, levels = CURRENT_TEAMS),
      toss_decision_factor = factor(toss_decision, levels = c("bat", "field")),
      winner_factor = factor(winner, levels = CURRENT_TEAMS)
    )
  
  # Feature engineering with more robust handling of NA values
  # Calculate team win rates
  team_stats <- data.frame()
  for (team in CURRENT_TEAMS) {
    team_matches <- sum(model_data$team1 == team | model_data$team2 == team, na.rm = TRUE)
    if (team_matches > 0) {
      team_wins <- sum((model_data$team1 == team & model_data$winner == team) | 
                      (model_data$team2 == team & model_data$winner == team), na.rm = TRUE)
      win_rate <- team_wins / team_matches
    } else {
      win_rate <- 0.5  # Default if no matches
    }
    
    team_stats <- rbind(team_stats, data.frame(team = team, win_rate = win_rate))
  }
  
  # Join team stats to model data
  model_data <- model_data %>%
    left_join(team_stats, by = c("team1" = "team")) %>%
    rename(team1_win_rate = win_rate) %>%
    left_join(team_stats, by = c("team2" = "team")) %>%
    rename(team2_win_rate = win_rate) %>%
    mutate(
      win_rate_diff = team1_win_rate - team2_win_rate,
      is_toss_winner_team1 = as.integer(toss_winner == team1),
      toss_advantage = as.integer(toss_winner == winner)
    )
  
  # Calculate venue advantage for each team
  venue_team_stats <- data.frame()
  for (team in CURRENT_TEAMS) {
    for (v in unique(model_data$venue)) {
      venue_matches <- sum((model_data$team1 == team | model_data$team2 == team) & 
                         model_data$venue == v, na.rm = TRUE)
      
      if (venue_matches > 0) {
        venue_wins <- sum(((model_data$team1 == team & model_data$winner == team) | 
                         (model_data$team2 == team & model_data$winner == team)) & 
                         model_data$venue == v, na.rm = TRUE)
        venue_win_rate <- venue_wins / venue_matches
      } else {
        venue_win_rate <- 0.5  # Default if no matches at this venue
      }
      
      venue_team_stats <- rbind(venue_team_stats, 
                              data.frame(team = team, venue = v, venue_advantage = venue_win_rate))
    }
  }
  
  # Join venue advantage to model data
  model_data <- model_data %>%
    left_join(venue_team_stats, by = c("team1" = "team", "venue" = "venue")) %>%
    rename(team1_venue_advantage = venue_advantage) %>%
    left_join(venue_team_stats, by = c("team2" = "team", "venue" = "venue")) %>%
    rename(team2_venue_advantage = venue_advantage) %>%
    mutate(
      team1_venue_advantage = coalesce(team1_venue_advantage, 0.5),
      team2_venue_advantage = coalesce(team2_venue_advantage, 0.5),
      venue_advantage_diff = team1_venue_advantage - team2_venue_advantage
    )
  
  # Calculate head-to-head for each team pair
  h2h_stats <- data.frame()
  for (team1 in CURRENT_TEAMS) {
    for (team2 in CURRENT_TEAMS) {
      if (team1 != team2) {
        h2h_matches <- model_data %>% 
          filter((team1 == team1 & team2 == team2) | 
                 (team1 == team2 & team2 == team1))
        
        if (nrow(h2h_matches) > 0) {
          h2h_wins <- sum(h2h_matches$winner == team1, na.rm = TRUE)
          h2h_rate <- h2h_wins / nrow(h2h_matches)
        } else {
          h2h_rate <- 0.5  # Default if no head-to-head matches
        }
        
        h2h_stats <- rbind(h2h_stats, 
                         data.frame(team1 = team1, team2 = team2, head_to_head = h2h_rate))
      }
    }
  }
  
  # Join head-to-head stats to model data
  model_data <- model_data %>%
    left_join(h2h_stats, by = c("team1", "team2")) %>%
    mutate(head_to_head = coalesce(head_to_head, 0.5))
  
  # Ensure no NA values in the features
  model_data <- model_data %>%
    mutate(across(where(is.numeric), ~coalesce(., 0.5)))
  
  # Select features for the model
  features <- model_data %>%
    select(
      team1_factor, team2_factor, venue_factor, toss_winner_factor, toss_decision_factor,
      team1_win_rate, team2_win_rate, win_rate_diff, is_toss_winner_team1,
      team1_venue_advantage, team2_venue_advantage, venue_advantage_diff, head_to_head,
      weight
    )
  
  # Target variable - binary outcome (1 if team1 wins, 0 if team2 wins)
  target <- as.integer(model_data$winner == model_data$team1)
  
  # Create training set
  set.seed(42)
  
  # Ensure no NA values in features
  features <- features %>%
    mutate(across(everything(), ~ifelse(is.na(.) | is.nan(.), 0.5, .)))
  
  # Remove any rows with NA in target
  valid_rows <- !is.na(target)
  features <- features[valid_rows, ]
  target <- target[valid_rows]
  weights <- features$weight
  features$weight <- NULL  # Remove weight from features
  
  train_idx <- createDataPartition(target, p = 0.8, list = FALSE)
  X_train <- features[train_idx, ]
  y_train <- target[train_idx]
  X_test <- features[-train_idx, ]
  y_test <- target[-train_idx]
  train_weights <- weights[train_idx]
  
  # Train random forest model
  rf_model <- randomForest(
    x = X_train,
    y = as.factor(y_train),
    ntree = 100,
    mtry = 4,
    importance = TRUE,
    na.action = na.omit,
    weights = train_weights
  )
  
  # Calculate evaluation metrics
  predictions <- predict(rf_model, X_test)
  actual <- as.factor(y_test)
  
  # Confusion Matrix
  conf_matrix <- table(Predicted = predictions, Actual = actual)
  
  # Extract values from confusion matrix
  TP <- conf_matrix[2, 2] # True Positives
  TN <- conf_matrix[1, 1] # True Negatives
  FP <- conf_matrix[2, 1] # False Positives
  FN <- conf_matrix[1, 2] # False Negatives
  
  # Calculate metrics
  accuracy <- (TP + TN) / sum(conf_matrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Print model evaluation
  cat("Model Training Complete\n")
  cat("Test Accuracy:", accuracy, "\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Store metrics for display in UI
  metrics <- list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    conf_matrix = conf_matrix
  )
  
  return(list(
    model = rf_model,
    data = ipl_data,
    metrics = metrics
  ))
}

# Train the model at startup
model_data <- train_model()
rf_model <- model_data$model
ipl_data <- model_data$data
model_metrics <- model_data$metrics

# Fix venue advantage calculation
venue_advantage <- model_data$data %>%
  group_by(venue, team1) %>%
  summarize(team1_venue_wins = sum(winner == team1, na.rm = TRUE),
            team1_venue_matches = n(),
            .groups = "drop") %>%
  mutate(team1_venue_advantage = team1_venue_wins / team1_venue_matches) %>%
  ungroup()

# Fix head-to-head records
head_to_head <- model_data$data %>%
  group_by(team1, team2) %>%
  summarize(t1_wins = sum(winner == team1, na.rm = TRUE),
            total = n(),
            .groups = "drop") %>%
  mutate(head_to_head = t1_wins / total) %>%
  ungroup()

# Define UI
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("IPL 2025 Match Winner Prediction"),
  
  tabsetPanel(
    tabPanel("Prediction",
      fluidRow(
        column(8, offset = 2,
          wellPanel(
            fluidRow(
              column(6, 
                selectInput("team1", "Team 1:", choices = c("Select Team 1" = "", CURRENT_TEAMS))
              ),
              column(6, 
                selectInput("team2", "Team 2:", choices = c("Select Team 2" = "", CURRENT_TEAMS))
              )
            ),
            fluidRow(
              column(6, 
                selectInput("venue", "Venue:", choices = c("Select Venue" = "", VENUES))
              ),
              column(6, 
                selectInput("toss_winner", "Toss Winner:", choices = c("Select Toss Winner" = ""))
              )
            ),
            fluidRow(
              column(6, 
                selectInput("toss_decision", "Toss Decision:", choices = c("Select" = "", "bat" = "bat", "field" = "field"))
              ),
              column(6, 
                actionButton("predict_btn", "Predict Winner", class = "btn-primary btn-block", style = "margin-top: 25px;")
              )
            )
          )
        )
      ),
      
      fluidRow(
        column(8, offset = 2,
          uiOutput("prediction_result")
        )
      )
    ),
    
    tabPanel("Model Performance",
      fluidRow(
        column(10, offset = 1,
          wellPanel(
            h3("Model Evaluation Metrics", style = "text-align: center;"),
            div(style = "display: flex; justify-content: space-around; margin-bottom: 20px;",
              div(
                h4("Accuracy"),
                div(style = "font-size: 24px; text-align: center; color: #18bc9c;", 
                    sprintf("%.2f%%", model_metrics$accuracy * 100))
              ),
              div(
                h4("Precision"),
                div(style = "font-size: 24px; text-align: center; color: #18bc9c;", 
                    sprintf("%.2f%%", model_metrics$precision * 100))
              ),
              div(
                h4("Recall"),
                div(style = "font-size: 24px; text-align: center; color: #18bc9c;", 
                    sprintf("%.2f%%", model_metrics$recall * 100))
              ),
              div(
                h4("F1 Score"),
                div(style = "font-size: 24px; text-align: center; color: #18bc9c;", 
                    sprintf("%.2f", model_metrics$f1_score))
              )
            ),
            h4("Confusion Matrix"),
            div(style = "display: flex; justify-content: center;",
              div(
                tags$table(style = "width: 400px; margin: 20px auto; border-collapse: collapse; border: 1px solid #ddd;",
                  tags$thead(
                    tags$tr(
                      tags$th(style = "padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;", ""),
                      tags$th(style = "padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;", "Actual: Team 1"),
                      tags$th(style = "padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;", "Actual: Team 2")
                    )
                  ),
                  tags$tbody(
                    tags$tr(
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5; font-weight: bold;", "Predicted: Team 1"),
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #dff0d8; text-align: center; font-weight: bold;", 
                              paste0(model_metrics$conf_matrix[1,1], " (True Negative)")),
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #f2dede; text-align: center;", 
                              paste0(model_metrics$conf_matrix[1,2], " (False Negative)"))
                    ),
                    tags$tr(
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5; font-weight: bold;", "Predicted: Team 2"),
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #f2dede; text-align: center;", 
                              paste0(model_metrics$conf_matrix[2,1], " (False Positive)")),
                      tags$td(style = "padding: 10px; border: 1px solid #ddd; background-color: #dff0d8; text-align: center; font-weight: bold;", 
                              paste0(model_metrics$conf_matrix[2,2], " (True Positive)"))
                    )
                  )
                )
              )
            ),
            hr(),
            div(
              h4("Interpretation"),
              p("Accuracy: Percentage of predictions that are correct."),
              p("Precision: When the model predicts Team 2 will win, how often is it right?"),
              p("Recall: Out of all actual Team 2 wins, how many did the model correctly identify?"),
              p("F1 Score: Harmonic mean of precision and recall. Higher values indicate better model performance."),
              h4("How to Read the Confusion Matrix"),
              p("True Negative (TN): Correctly predicted that Team 1 would win."),
              p("False Negative (FN): Incorrectly predicted that Team 1 would win when Team 2 actually won."),
              p("False Positive (FP): Incorrectly predicted that Team 2 would win when Team 1 actually won."),
              p("True Positive (TP): Correctly predicted that Team 2 would win.")
            )
          )
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  # Update toss winner choices based on team selections
  observe({
    teams <- c()
    if (input$team1 != "") teams <- c(teams, input$team1)
    if (input$team2 != "") teams <- c(teams, input$team2)
    
    updateSelectInput(session, "toss_winner",
                     choices = c("Select Toss Winner" = "", teams))
  })
  
  # Validate that teams are different
  team_error <- reactive({
    if (input$team1 != "" && input$team2 != "" && input$team1 == input$team2) {
      return("Please select different teams")
    }
    return(NULL)
  })
  
  # Make prediction when button is clicked
  prediction_data <- eventReactive(input$predict_btn, {
    # Validate inputs
    req(input$team1, input$team2, input$venue, input$toss_winner, input$toss_decision)
    
    if (!is.null(team_error())) {
      return(list(error = team_error()))
    }
    
    # Prepare data for prediction
    team1 <- input$team1
    team2 <- input$team2
    venue <- input$venue
    city <- venue_city_map[[venue]]
    toss_winner <- input$toss_winner
    toss_decision <- input$toss_decision
    
    # Get historical win rates
    team1_matches <- sum(ipl_data$team1 == team1 | ipl_data$team2 == team1, na.rm = TRUE)
    team2_matches <- sum(ipl_data$team1 == team2 | ipl_data$team2 == team2, na.rm = TRUE)
    
    team1_wins <- sum((ipl_data$team1 == team1 & ipl_data$winner == team1) | 
                     (ipl_data$team2 == team1 & ipl_data$winner == team1), na.rm = TRUE)
    team2_wins <- sum((ipl_data$team1 == team2 & ipl_data$winner == team2) | 
                     (ipl_data$team2 == team2 & ipl_data$winner == team2), na.rm = TRUE)
    
    team1_win_rate <- ifelse(team1_matches > 0, team1_wins / team1_matches, 0.5)
    team2_win_rate <- ifelse(team2_matches > 0, team2_wins / team2_matches, 0.5)
    
    # Get venue advantage
    team1_venue_matches <- sum((ipl_data$team1 == team1 | ipl_data$team2 == team1) & 
                             ipl_data$venue == venue, na.rm = TRUE)
    team2_venue_matches <- sum((ipl_data$team1 == team2 | ipl_data$team2 == team2) & 
                             ipl_data$venue == venue, na.rm = TRUE)
    
    team1_venue_wins <- sum(((ipl_data$team1 == team1 & ipl_data$winner == team1) | 
                          (ipl_data$team2 == team1 & ipl_data$winner == team1)) & 
                          ipl_data$venue == venue, na.rm = TRUE)
    team2_venue_wins <- sum(((ipl_data$team1 == team2 & ipl_data$winner == team2) | 
                          (ipl_data$team2 == team2 & ipl_data$winner == team2)) & 
                          ipl_data$venue == venue, na.rm = TRUE)
    
    team1_venue_adv <- ifelse(team1_venue_matches > 0, team1_venue_wins / team1_venue_matches, 0.5)
    team2_venue_adv <- ifelse(team2_venue_matches > 0, team2_venue_wins / team2_venue_matches, 0.5)
    
    # Calculate head-to-head
    head_to_head_matches <- ipl_data %>% 
      filter((team1 == input$team1 & team2 == input$team2) | 
             (team1 == input$team2 & team2 == input$team1))
    
    if (nrow(head_to_head_matches) > 0) {
      team1_h2h_wins <- sum(head_to_head_matches$winner == input$team1, na.rm = TRUE)
      head_to_head <- team1_h2h_wins / nrow(head_to_head_matches)
    } else {
      head_to_head <- 0.5
    }
    
    # Try to make a prediction
    prediction_result <- tryCatch({
      # Create prediction features using factors with the same levels as the model
      pred_data <- data.frame(
        team1_win_rate = team1_win_rate,
        team2_win_rate = team2_win_rate,
        win_rate_diff = team1_win_rate - team2_win_rate,
        is_toss_winner_team1 = as.integer(toss_winner == team1),
        team1_venue_advantage = team1_venue_adv,
        team2_venue_advantage = team2_venue_adv,
        venue_advantage_diff = team1_venue_adv - team2_venue_adv,
        head_to_head = head_to_head,
        stringsAsFactors = FALSE
      )
      
      # Add factor variables using the same structure as the model
      pred_data$team1_factor <- factor(team1, levels = levels(rf_model$forest$xlevels$team1_factor))
      pred_data$team2_factor <- factor(team2, levels = levels(rf_model$forest$xlevels$team2_factor))
      pred_data$venue_factor <- factor(venue, levels = levels(rf_model$forest$xlevels$venue_factor))
      pred_data$toss_winner_factor <- factor(toss_winner, levels = levels(rf_model$forest$xlevels$toss_winner_factor))
      pred_data$toss_decision_factor <- factor(toss_decision, levels = levels(rf_model$forest$xlevels$toss_decision_factor))
      
      # Make prediction
      win_prob <- predict(rf_model, pred_data, type = "prob")
      team1_prob <- win_prob[1, "1"] 
      team2_prob <- win_prob[1, "0"]
      
      # Determine winner
      if (team1_prob > team2_prob) {
        winner <- team1
        win_pct <- team1_prob * 100
      } else {
        winner <- team2
        win_pct <- team2_prob * 100
      }
      
      # Return all required values in a list
      list(
        winner = winner,
        win_pct = win_pct,
        team1_prob = team1_prob * 100,
        team2_prob = team2_prob * 100
      )
    }, error = function(e) {
      # Fallback to simple heuristic if model prediction fails
      team1_score <- team1_win_rate + (team1_venue_adv * 0.5) + (head_to_head * 0.5) + 
                   (if(toss_winner == team1) 0.1 else 0)
      team2_score <- team2_win_rate + (team2_venue_adv * 0.5) + ((1-head_to_head) * 0.5) + 
                   (if(toss_winner == team2) 0.1 else 0)
      
      local_winner <- NA
      local_win_pct <- NA
      local_team1_prob <- NA
      local_team2_prob <- NA
      
      if (team1_score > team2_score) {
        local_winner <- team1
        # Convert to percentage (rough estimate)
        local_win_pct <- 50 + (team1_score - team2_score) * 50
        local_win_pct <- min(local_win_pct, 95) # Cap at 95%
        local_team1_prob <- local_win_pct
        local_team2_prob <- 100 - local_team1_prob
      } else {
        local_winner <- team2
        local_win_pct <- 50 + (team2_score - team1_score) * 50
        local_win_pct <- min(local_win_pct, 95) # Cap at 95%
        local_team2_prob <- local_win_pct
        local_team1_prob <- 100 - local_team2_prob
      }
      
      # Return values from error handler
      list(
        winner = local_winner,
        win_pct = local_win_pct,
        team1_prob = local_team1_prob,
        team2_prob = local_team2_prob
      )
    })
    
    # Extract values from the prediction result
    winner <- prediction_result$winner
    win_pct <- prediction_result$win_pct
    team1_prob <- prediction_result$team1_prob
    team2_prob <- prediction_result$team2_prob
    
    # Check for NA or NULL values
    if (is.na(winner) || is.null(winner)) {
      return(list(error = "Could not make a prediction. Please try different teams or venues."))
    }
    
    # Ensure all values are proper
    win_pct <- ifelse(is.na(win_pct) || is.null(win_pct), 50, win_pct)
    team1_prob <- ifelse(is.na(team1_prob) || is.null(team1_prob), 50, team1_prob)
    team2_prob <- ifelse(is.na(team2_prob) || is.null(team2_prob), 50, team2_prob)
    
    # Generate explanation
    reasons <- c()
    
    # Check historical performance
    if (winner == team1 && team1_win_rate > team2_win_rate) {
      reasons <- c(reasons, paste0(winner, " has a better overall winning record in the tournament"))
    } else if (winner == team2 && team2_win_rate > team1_win_rate) {
      reasons <- c(reasons, paste0(winner, " has a better overall winning record in the tournament"))
    }
    
    # Check head to head
    if (winner == team1 && head_to_head > 0.5) {
      reasons <- c(reasons, paste0(winner, " has a stronger head-to-head record against ", team2))
    } else if (winner == team2 && head_to_head < 0.5) {
      reasons <- c(reasons, paste0(winner, " has a stronger head-to-head record against ", team1))
    }
    
    # Check venue advantage
    if (winner == team1 && team1_venue_adv > team2_venue_adv) {
      reasons <- c(reasons, paste0(winner, " has historically performed better at ", venue))
    } else if (winner == team2 && team2_venue_adv > team1_venue_adv) {
      reasons <- c(reasons, paste0(winner, " has historically performed better at ", venue))
    }
    
    # Check toss advantage
    if (toss_winner == winner) {
      reasons <- c(reasons, paste0("Winning the toss and choosing to ", toss_decision, 
                                  " gives ", winner, " an advantage"))
    }
    
    # If no specific reasons found, add a general one
    if (length(reasons) == 0) {
      reasons <- c("Based on the overall statistical analysis of past performance")
    }
    
    explanation <- paste("This prediction is based on several factors: ", 
                        paste(reasons, collapse = ", "), ".", sep = "")
    
    return(list(
      team1 = team1,
      team2 = team2,
      team1_prob = team1_prob,
      team2_prob = team2_prob,
      winner = winner,
      win_pct = win_pct,
      explanation = explanation
    ))
  })
  
  # Render prediction result
  output$prediction_result <- renderUI({
    result <- tryCatch({
      prediction_data()
    }, error = function(e) {
      return(list(error = paste("Error in prediction:", e$message)))
    })
    
    if (is.null(result)) {
      return(NULL)
    }
    
    if (!is.null(result$error)) {
      div(
        class = "alert alert-danger",
        result$error
      )
    } else {
      tryCatch({
        # Check if all required fields are available
        if (is.null(result$winner) || is.null(result$win_pct) || 
            is.null(result$team1_prob) || is.null(result$team2_prob)) {
          return(div(
            class = "alert alert-danger",
            "Missing prediction data. Please try again with different inputs."
          ))
        }
        
        div(
          class = "well prediction-result",
          h3("Predicted Winner: ", 
            span(result$winner, style = "font-weight: bold; color: #2c3e50;")
          ),
          div(class = "win-chance", 
            "Win Chance: ", 
            span(sprintf("%.1f%%", result$win_pct), 
                style = "font-weight: bold; color: #18bc9c;")
          ),
          p(class = "explanation", result$explanation),
          hr(),
          fluidRow(
            column(6, 
              div(style = "text-align: center; margin-bottom: 10px;", 
                strong(result$team1)
              ),
              div(style = paste0("background-color: #3498db; height: 20px; width: ", 
                              min(max(result$team1_prob, 5), 100), "%; margin: 0 auto;")),
              div(style = "text-align: center; margin-top: 5px;", 
                paste0(sprintf("%.1f", result$team1_prob), "%")
              )
            ),
            column(6,
              div(style = "text-align: center; margin-bottom: 10px;", 
                strong(result$team2)
              ),
              div(style = paste0("background-color: #e74c3c; height: 20px; width: ", 
                              min(max(result$team2_prob, 5), 100), "%; margin: 0 auto;")),
              div(style = "text-align: center; margin-top: 5px;", 
                paste0(sprintf("%.1f", result$team2_prob), "%")
              )
            )
          )
        )
      }, error = function(e) {
        div(
          class = "alert alert-warning",
          h4("Simplified Prediction"),
          p("Due to limited data, we're using a simplified prediction model."),
          div(
            h3("Predicted Winner: ", 
              span(result$winner, style = "font-weight: bold; color: #2c3e50;")
            ),
            div(class = "win-chance", 
              "Win Chance: ", 
              span(sprintf("%.1f%%", result$win_pct), 
                  style = "font-weight: bold; color: #18bc9c;")
            ),
            p(class = "explanation", ifelse(is.null(result$explanation), 
                                          "Based on historical performance and match conditions.", 
                                          result$explanation))
          )
        )
      })
    }
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server) 