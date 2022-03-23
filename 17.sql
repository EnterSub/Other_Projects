select distinct Product.type, Product.model, Laptop.speed from Product
join Laptop
on Product.model = Laptop.model
where Product.type = 'Laptop'
and Laptop.speed < (select min(speed) from PC)

-- Find notebook PC models that are slower than the speed of each of the PCs.
-- Output: type, model, speed
