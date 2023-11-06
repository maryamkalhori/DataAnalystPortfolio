/*
➕Table 1: Customers
Columns: customer_id (INT), name (VARCHAR(50)), email (VARCHAR(50)), phone_number (VARCHAR(20))
Primary Key: customer_id

➕Table 2: Orders
Columns: order_id (INT), customer_id (INT), order_date (DATE)
Primary Key: order_id
Foreign Key: customer_id references Customers(customer_id)

➕Table 3: OrderItems
Columns: order_item_id (INT), order_id (INT), product_id (INT), quantity (INT)
Primary Key: order_item_id
Foreign Key: order_id references Orders(order_id)


➕Table 4: Products
Columns: id (INT), name (VARCHAR(50)), category (VARCHAR(50)), price (DECIMAL(10,2))
Primary Key: id


The script also includes code to generate random data and fill the tables:
⛔A table called "IDs" is created to generate a list of row IDs.
⛔The "Customers" table is filled with 5000 rows of randomly generated customer data.
⛔The "Orders" table is filled with 35000 rows of randomly generated order data.
⛔The "Products" table is filled with 20 rows of predefined product data.
⛔The "OrderItems" table is filled with a random number of rows (between 1 and 10) for each order, linking to random products.
*/


--Table 1: Customers
CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(50),
    email VARCHAR(50),
    phone_number VARCHAR(20)
);

--Table 2: Orders
CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE
);

--Table 3: OrderItems
CREATE TABLE OrderItems (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id)
);
--Table 4: Products
CREATE TABLE product (
id INT PRIMARY KEY,
name VARCHAR(50),
category VARCHAR(50),
price DECIMAL(10,2)
);
---------------------------------------------------------------------
--Fill tables with random data
--------------------------------------------------------------------
--First we generate ID table
CREATE TABLE IDs (
    row_id INTEGER PRIMARY KEY
);
WITH RECURSIVE
cte_row_generator(row_id) AS (
    VALUES (1)
    UNION ALL
    SELECT row_id + 1 FROM cte_row_generator WHERE row_id < 1000000
)
INSERT INTO IDs (row_id)
SELECT row_id FROM cte_row_generator;
----------------------------------------------------------------------

INSERT INTO Customers (customer_id, name, email, phone_number)
SELECT 
    ROW_NUMBER() OVER (ORDER BY random()) AS customer_id,
    'Customer' || ROW_NUMBER() OVER (ORDER BY random()) AS name,
    'customer' || ROW_NUMBER() OVER (ORDER BY random()) || '@example.com' AS email,
    '555-' || substr('0000' || CAST((random() + 1000) AS INTEGER), -4) AS phone_number
FROM 
    IDs AS c1
LIMIT 5000;
----------------------------------------------------------------------
INSERT INTO Orders (order_id, customer_id, order_date)
SELECT 
ROW_NUMBER() OVER (ORDER BY random()) AS order_id,
abs(random() % 5000) customer_id, 
IFNULL(date('now', '-' || CAST(FLOOR(random() % 365) + 1 AS TEXT) || ' days'),
date('now', '-' || CAST(FLOOR(random() % 365) + 1 AS TEXT) || ' days')) as order_date
FROM IDs
limit 35000;
---------------------------------------------------------------------
INSERT INTO Products (id, name, category, price) VALUES
(1, 'Milk', 'Groceries', 2.99),
(2, 'Eggs', 'Groceries', 3.49),
(3, 'Bread', 'Groceries', 1.99),
(4, 'Cheese', 'Groceries', 4.99),
(5, 'Butter', 'Groceries', 2.49),
(6, 'Apple', 'Groceries', 0.99),
(7, 'Banana', 'Groceries', 0.79),
(8, 'Orange', 'Groceries', 1.29),
(9, 'Coffee', 'Groceries', 5.99),
(10, 'Tea', 'Groceries', 4.49),
(11, 'Shampoo', 'Personal Care', 6.99),
(12, 'Soap', 'Personal Care', 2.99),
(13, 'Toothpaste', 'Personal Care', 3.99),
(14, 'Toothbrush', 'Personal Care', 1.99),
(15, 'Deodorant', 'Personal Care', 4.99),
(16, 'Razor', 'Personal Care', 5.99),
(17, 'Shaving Cream', 'Personal Care', 3.99),
(18, 'Lotion', 'Personal Care', 7.99),
(19, 'Towel', 'Personal Care', 9.99),
(20, 'Hair Dryer', 'Personal Care', 19.99);
--------------------------------------------------------------------------
WITH numbers AS (
SELECT 1 AS num
UNION ALL
SELECT num + 1 FROM numbers WHERE num <= 10
)
INSERT INTO OrderItems (order_item_id, order_id, product_id, quantity)
SELECT 
    ROW_NUMBER() OVER (ORDER BY random()) AS order_item_id,
    o.order_id,
    abs(random() % 20) + 1 AS product_id,
    abs(random() % 10) + 1 AS quantity
FROM orders o
cross JOIN numbers n 
where 
n.num <= abs(round(random()) % 10) + 1
