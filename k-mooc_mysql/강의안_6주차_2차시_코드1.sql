-- ## ��¥�� ����� ��� ���ž��� �����ϴ� ���� ##
SELECT dt
	 , COUNT(*) AS purchase_count
	 , SUM(purchase_amount) AS total_amount
	 , AVG(purchase_amount * 1.0) AS avg_amount
FROM purchase_log
GROUP BY dt
ORDER BY dt;