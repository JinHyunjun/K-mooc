-- ## ���� �ڷ����� �����͸� ������ ���� ##
SELECT dt
	 , ad_id
	 , clicks / impressions AS ctr_int
	 , 1.0 * clicks / impressions AS ctr_float
	 , 100.0 * clicks / impressions AS ctr_as_percent
FROM advertising_stats
WHERE dt = '2017-04-01'
ORDER BY dt, ad_id;