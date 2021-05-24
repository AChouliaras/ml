function [F1,precision,recall,accuracy] = printClassMetrics (pred_val , yval, verbose)

  accuracy = mean(double(pred_val == yval))
  acc_all0 = mean(double(0 == yval))

  actual_positives = sum(yval == 1);
  actual_negatives = sum(yval == 0);
  true_positives = sum((pred_val == 1) & (yval == 1));
  false_positives = sum((pred_val == 1) & (yval == 0));
  false_negatives = sum((pred_val == 0) & (yval == 1));
  precision = 0; 
  if ( (true_positives + false_positives) > 0)
    precision = true_positives / (true_positives + false_positives);
  end 

  recall = 0; 
  if ( (true_positives + false_negatives) > 0 )
    recall = true_positives / (true_positives + false_negatives);
  end 

  F1 = 0; 
  if ( (precision + recall) > 0) 
    F1 = 2 * precision * recall / (precision + recall);
  end
 
  if (verbose) 
    true_positives
    actual_positives
    false_positives
    false_negatives
    precision
    recall
    F1
  end 
  
end
