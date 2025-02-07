# 1) Find Customers Who Never Order (LeetCode #183)

SELECT c.name 
FROM Customers c 
LEFT JOIN Orders o ON c.id = o.customer_id 
WHERE o.id IS NULL;

# 2) Consecutive Numbers (LeetCode #180)

SELECT DISTINCT num 
FROM (
    SELECT num, 
           LEAD(num) OVER() AS next_num, 
           LEAD(num, 2) OVER() AS next_next_num 
    FROM Logs
) subquery
WHERE num = next_num AND num = next_next_num;