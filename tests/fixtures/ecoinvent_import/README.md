# Invented EcoSpold import inventories

Two invented inventories exercise nonunit production, signed waste demand,
linked elementary flows, product metadata and lognormal uncertainty through
real bw2io import and Brightway persistence. They contain no licensed data.

The product produces 2 kg, emits 3 kg of the fixture flow and demands -0.5 kg
of waste treatment. Treatment produces -4 kg and emits -0.25 kg of that flow.
One kg of product therefore has the inventory
`(3 + (-0.5/-4)*(-0.25))/2 = 1.484375 kg`. The test independently solves this
matrix and checks every imported exchange, both reference bases and source
custody. These quantities test the import boundary; they are not LCI sources.

With `bw2io==0.9.17` and `reparametrize_lognormals=False`, the supported import
strategy sets lognormal `loc=log(abs(amount))`. The product's 3 kg emission
therefore has `loc=log(3)` and its original scale 0.2. This conversion follows bw2io 0.9.17's import of the EcoSpold uncertainty
parameters.
