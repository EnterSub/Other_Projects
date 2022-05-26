with table_1 as (select count(model) as count_model, count(maker) as count_table from Product
group by maker
having count(model) = 1)
select count(count_model) from table_1

-- Using the Product table, determine the number of manufacturers producing one model.
