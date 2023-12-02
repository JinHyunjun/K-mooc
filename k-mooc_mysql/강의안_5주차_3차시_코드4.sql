-- ## 0���� ������ ���� ���� CTR�� ����ϴ� ���� ##
SELECT dt
	 , ad_id
	-- 0���� ������ ���� �߻�
	-- 100.0 * clicks / impressions AS ctr_as_percent
	 , CASE WHEN impressions > 0 THEN 100.0 * clicks / impressions END AS ctr_as_percent_by_case
FROM advertising_stats
ORDER BY dt, ad_id;