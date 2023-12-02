-- ## ROLLUP�� ����ؼ� ī�װ��� ����� �Ұ踦 ���ÿ� ���ϴ� ���� ##
SELECT COALESCE(category, 'all') AS category
	 , COALESCE(sub_category, 'all') AS sub_category
	 , SUM(price) AS amount
FROM purchase_detail_log
GROUP BY ROLLUP(category, sub_category);
-- < Hive >
-- GROUP BY category, sub_category WITH ROLLUP