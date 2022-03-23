Select distinct Product.model, price from Product
join PC
on Product.model = PC.model
where maker = 'B'
union
Select distinct Product.model, price from Product
join Laptop
on Product.model = Laptop.model
where maker = 'B'
union
Select distinct Product.model, price from Product
join Printer
on Product.model = Printer.model
where maker = 'B'

-- Find the model numbers and prices of all commercially available products (of any type) from manufacturer B (Latin letter).
