### churn
>customers who left within last month 
### why churn prediction matters ? 
> Customer retention saves more money than new customer aqcuisition
### what decision this model will support ?
> who to target as churn

----
## Churn is imbalanced : ~26% churn rate 
### Accuracy alone is not the metric : need ROC/AUC , Precision/recall

## Tenure of churn=yes is lower , that means customer with less tenure are likely to leave 
## little overlap , of two boxes , overlap btw 15 to 29 months 
## loyal customers stay for longer durations
## implies tenure has strong relation with churn  (warning~leakage possible )

## Churn cannot occur for tenure = 0 becuase the customer did not have a tenure before , hence churn =  NO for tenure = 0 . 

## boxes overlap very much (56.15 to 88.4)
## median shows that churners have higher cost than retainers 
## also most of the retainers have low cost than churners as per the boxes 
## this implies that i just cannot only consider monthlyprice as the only feature that shows if its a churner 


## i can see that monthlycharges and total charges have diagonal relationship for churn = no 
## and for churn = yes i dont see that much of it and density is more towaredds x axis horizontaly
## this gives that totalCharge is actually telling us the tenure and retentiont time 

## Churn rate decreases as contract lenght increases , it is a reliable feature 

## churn rate drops significantly from 0 service to 1 service . then from 1 to 3 service the churn rate increases and from 3 it decreases. it also tells that having a higher number of services like 7,8 will result in lower churn rate

## Service-specific features contribute unequally to churn prediction and should be weighted accordingly during modeling.

## Customers with partners or dependant show lower churn rate . Weak to moderate relationship 

### tenure and total charges encode survival time and may partially leak outcome information