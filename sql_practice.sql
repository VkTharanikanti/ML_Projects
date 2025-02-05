CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    joining_date DATE
);
INSERT INTO employees (id, name, department, salary, joining_date) VALUES 
(1, 'Alice', 'HR', 55000, '2022-03-01'),
(2, 'Bob', 'IT', 72000, '2021-06-15'),
(3, 'Charlie', 'Finance', 65000, '2020-09-23'),
(4, 'David', 'IT', 80000, '2019-12-11'),
(5, 'Emma', 'Marketing', 60000, '2023-01-20');

-- Query 1: Retrieve all employees from the IT department

SELECT *
	FROM employees
	WHERE department = 'IT'
;

-- Query 2: Find the average salary of all employees

SELECT AVG(salary) as avg_sal
	FROM employees
;

-- Query 3: Get the highest-paid employee

SELECT name, salary
	FROM employee
	ORDERBY salary DESC
	limit 1
;

-- Query 4: Count employees in each department

SELECT department, count(*) AS Total_emp
	FROM employee
	GROUPBY department
;

-- Query 5: Get employees who joined after Jan 1, 2021

SELECT *
	FROM employee
	WHERE joining_date > '2021-01-01'
;