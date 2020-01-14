# Ports

## Input

A .csv file.

> vector [id, band0, band2, ... band24, sourceid]

> vector1, vector2,...

band* means:

> The spectral range is from 480.9 to 946.8nm with 32 continuous bands (Wu et al., 2013). Only Bands 6–9 (522.4–606.0nm) and 11–31 (631.2–918.1nm) are used due to the relatively high signal to noise ratios (SNRs). Therefore 25 bands are finally used in this work . 

\* from 0 to 24

## Output

a .csv file

> vector [id, w0, w1, ... w6]

> vector1, vector2, vector3, ...

w* means:

>  SiO2 (wt%) Al2O3 (wt%) CaO(wt%) FeO(wt%) MgO(wt%) TiO2 (wt%) Mg#(×10−2) 

\* from 0 to 6

For every set, one example has an id, only one. For the same example, id in Input.csv and Output.csv is the same.

## Source list

* 0 test
