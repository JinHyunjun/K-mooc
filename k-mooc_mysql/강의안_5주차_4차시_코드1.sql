-- ## �ڵ带 ���̺�� �����ϴ� ���� ##
SELECT user_id
	 , CASE WHEN register_device = 1 THEN '����ũ��'
			WHEN register_device = 2 THEN '����Ʈ��'
			WHEN register_device = 3 THEN '���ø����̼�'
			ELSE '' END AS device_name
FROM mst_users;