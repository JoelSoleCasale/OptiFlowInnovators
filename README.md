<p align="center">
  <img width="425" height="193" src="project_dashboard/LogoOptiFlow.jpeg">
</p>

<h1 align="center">
    Optimizing Supply Chain orders for storable products under
unified demand <br>
</h1>


## Abstract
In this project, we propose a mathematical model and implementation based on a collaborative scheme designed to optimize the storage and distribution of medical products to hospitals given historical data.

## Challenge

*What is expected from the challenge?*
- 


## Evaluation Criteria
*What will be measured?*
- **Answer to the business case**: 

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



## Other parts of our project