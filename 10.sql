select Product.model, price from Product
join Printer
on Product.model = Printer.model
where price = (select max(price) from Printer)

-- Find the highest priced printer models. Output: model, price
