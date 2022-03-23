Select distinct maker from Product
join PC
on Product.model = PC.model
where speed >= 750
intersect
select distinct maker from Product
join Laptop
on Product.model = Laptop.model
where speed >= 750

-- Find manufacturers that produce like PCs
-- with a speed of at least 750 MHz, and PC notebooks with a speed of at least 
-- 750 MHz.
-- Display: Maker
