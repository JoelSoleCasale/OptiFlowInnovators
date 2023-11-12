<p align="center">
  <img width="425" height="193" src="project_dashboard/LogoOptiFlow.jpeg">
</p>

<h1 align="center">
    Streamlining Inventory Management for Storable Healthcare Products in a Unified Demand Environment <br>
</h1>


## Abstract
In this project, we propose a mathematical model and implementation based on a collaborative scheme designed to optimize the storage and distribution of medical products to hospitals given historical data.

## Challenge

**MAIN GOAL**: Stablish a purchase plan for 2023

*What else can we analyse?*
- Predict the expenses for the following year by product
- Detect purchase patterns and combinations
- Evaluate the environmental impact if purchase strategy changes


### Evaluation Criteria
*What will be measured?*
- **Answer to the business case**

- **Analytical skills**

- **Storytelling & rationale**

- **Innovation**

- **Oral presentation skills**

## Data
Some information about the given data:

### consumo_material_clean.xlsx
Purchase history since 2015 for of all the healthcare supplies of a group of hospitals
- `CODIGO` *(object)*: Product code (first letter relates to product classification). Unique identifier of a product
- `PRODUCTO` *(object)*: Product description
- `FECHAPEDIDO` *(Date)*: Purchase date (day/month/year) 
- `NUMERO` *(object)*: Order number/year 
- `REFERENCIA` *(object)*: Reference number
- `CANTIDADCOMPRA` *(Numeric)*: Number of products purchased 
- `UNIDADESCONSUMOCONTENIDAS` *(Numeric)*: Number of units that the product contains
- `PRECIO` *(Numeric)*: Cost in €
- `IMPORTELINEA` *(Numeric)*: Total cost of products purchase
- `TIPOCOMPRA` *(object)*: Type of public purchase (Compra menor: minor contract / Concurso : public tender) 
- `ORIGEN` *(object)*: Code corresponding to the purchasing region-hospital-department (anonymized data)
- `TGL` *(object)*: Type of logistic distribution of products (transito: directly delivered at the hospital /almacenable: delivered at purchase center) 



## Solution 
*(See paper for more information)*

### Introduction
Supply chain robustness is of paramount importance for businesses, as it not only ensures operational continuity but also provides significant financial benefits. Especially in medical settings, robustness is imperial as the effects of a shortage can cost human lives. Additionally, it is of paramount importance for a company to minimize the ecological footprint stemming from its activities, especially related to transportation which is a major source of pollution, 28% as of 2021.

### Our approach: Problem Statement
The goal is therefore to find cost-optimality whilst satisfying robustness and environmental constraints. We suggest a unified demand approach to tackle the problem, that is an agreement reached by an array of medical institutions to combine the demand for medical products in one single order to reduce cost, followed by a mathematical model and further implementation to solve the problem in this setting. For simplicity, we consider a time granularity of months.

<p align="center">
  <img width="300" height="225" src="./Paper/NTTStorableSupplyPlanner/WhatsApp Image 2023-11-12 at 01.28.21.jpeg">
  <figcaption>Figure 1 - Image showing concatenation of 3 models (3 products), assuming that the distri- bution centers for each product are in the same location (graphical purposes).</figcaption>
</p>

### Proposed solution
In the unified demand scenario, we assume that all the hospitals group their orders for a given product in one order, with the referenced economical benefits that this supposes. We consider therefore each product separately and build a model to optimize the processes for each product.

Given a specific product, we consider a unique provider and distribution center (as we are in a given region we can assume a certain locality). Our challenge is to optimize the costs given a certain environmental footprint and resilience score.
We quantify the environmental impact as proportional to the number of orders, as referenced in and in the problem specification. This amounts to choosing when and how much to or- der given robust satisfaction of demand, fixed number of orders, and storage costs.

The demand we obtain comes from a prediction for the purchase plan, from which we yield the amount of units needed for the coming year.