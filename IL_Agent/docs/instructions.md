  How to run                                                                                                                                                         
                                                                                                                                                                     
  Step 1 — train models (if not already trained):                                                                                                                    
  cd IL_Agent && python src/training.py                                                                                                                              
  This reads parser/cleaned_dataset.jsonl and saves models/move_model.keras + models/switch_model.keras.                                                             
                                                                                                                                                                     
  Step 2a — play vs human:                                                                                                                                           
  cd IL_Agent && python main.py vs_human                                                                                                                             
  Challenge IL_Agent_1 on the server.                                                                                                                                
                                                                                                                                                                     
  Step 2b — co-training loop (vs RL agent):                                                                                                                          
  # Terminal 1                                                                                                                                                       
  cd IL_Agent && python main.py vs_rl                                                                                                                                
                                                                                                                                                                     
  # Terminal 2                                                                                                                                                       
  cd RL_Agent && python main.py vs_il                                                                                                                                
                                                                                                                                                                     
  The IL agent plays random until models are trained. Once models exist, it uses the Keras predictions with action masking and falls back to random on any inference
  error.             