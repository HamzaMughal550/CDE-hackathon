use bangood;

select * from banggood;

-- 1. Average price per category
SELECT category, AVG(CAST(price_numeric AS FLOAT)) AS avg_price
FROM banggood
GROUP BY category;

-- 2. Average rating per category
SELECT category, AVG(CAST(rating_numeric AS FLOAT)) AS avg_rating
FROM banggood
GROUP BY category;

-- 3. Product count per category
SELECT category, COUNT(*) AS product_count
FROM banggood
GROUP BY category;

-- 4. Top 5 reviewed items per category
SELECT category, name, review_count
FROM banggood
ORDER BY CAST(review_count AS INT) DESC
-- Optionally, filter by category using WHERE
-- For each category, you can use ROW_NUMBER() OVER (PARTITION BY category ORDER BY CAST(review_count AS INT) DESC)
;
--Average popularity score per category
SELECT category, AVG(CAST(popularity_score AS FLOAT)) AS avg_popularity
FROM banggood
GROUP BY category;

