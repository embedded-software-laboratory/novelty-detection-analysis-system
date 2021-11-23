delimiter //
create procedure dataDensity(pLimit int)
begin
	declare n int;
	declare i int;
	declare pid int;
	create temporary table res(patientid int, count int);
	set n = (select count(*) from (select distinct patientid from SMITH_ASIC_SCHEME.$placeholder ad) as patientids);
	set i = 0;
	while i<n do
		set pid = (select distinct patientid from SMITH_ASIC_SCHEME.$placeholder ad2 limit i,1);
		insert into res select patientid, count(*) from SMITH_ASIC_SCHEME.$placeholder ad where patientid = pid;
		set i = i+1;
	end while;
	select * from res order by count desc limit pLimit;
	drop table res;
end//