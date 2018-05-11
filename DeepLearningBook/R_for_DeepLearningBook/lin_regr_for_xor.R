# If you were not install tidyverse,
# install.packages("tidyverse")
library(tidyverse)

input <- tibble(
  x1 = c(TRUE, TRUE, FALSE, FALSE),
  x2 = c(TRUE, FALSE, TRUE, FALSE)
)

output <- tibble(
  y = c(FALSE, TRUE, TRUE, FALSE)
)

regr <- lm(output$y ~ input$x1 + input$x2)
broom::tidy(regr) %>%
  knitr::kable(.)

modelr::add_predictions(input, regr)
