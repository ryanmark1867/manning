## happy path
* greet
  - utter_greet
* predict_delay
  - action_predict_delay
 
## New Story
* predict_delay_complete{"route":"501","direction":"e"}
  - slot{"route":"501"}
  - slot{"direction":"e"}
  - action_predict_delay_complete

## New Story
* predict_delay_complete{"route":"506","hour":"16","day":"today"}
  - slot{"route":"506"}
  - slot{"hour":"16"}
  - slot{"day":"today"}
  - action_predict_delay_complete


## New Story
* predict_delay_complete{"route":"508","delta":"3","scale":"hours"}
  - slot{"route":"508"}
  - slot{"delta":"3"}
  - slot{"scale":"hours"}
  - action_predict_delay_complete

## say goodbye
* goodbye
  - utter_goodbye
