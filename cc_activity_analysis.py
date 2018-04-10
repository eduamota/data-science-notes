# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:27:04 2018

@author: emota
"""

#%%

import pandas as pd
import numpy as np

#%%

'''select c.*, convert_tz(from_unixtime(c.entered_timestamp), 'UTC', 'America/Vancouver') as date_time from  `call` c
where convert_tz(from_unixtime(c.entered_timestamp), 'UTC', 'America/Vancouver') between '2018-01-01' and '2018-04-09'
'''

call_dataset = pd.read_csv("C:\\Users\\emota\\Documents\DataScience\\calls_apr_8.csv")

call_dataset.head()

#%%

'''
select c.*, convert_tz(c.Start_Time, 'UTC', 'America/Vancouver') as date_time from  `agent_status` c
where convert_tz(c.Start_Time, 'UTC', 'America/Vancouver') between '2018-01-01' and '2018-04-09'
'''

agent_dataset = pd.read_csv("C:\\Users\\emota\\Documents\DataScience\\all_agents_apr_8.csv")

agent_dataset.head()

#%%

aht_mask = agent_dataset['State'] == 'connected'


#%%
handled = agent_dataset.loc[aht_mask,:]

#%%

handled.columns

#%%

handled['Duration'].fillna(handled['Duration'])

#%%

handled['Duration'].replace('\\0', None, inplace=True)


#%%

handled['Duration'] = handled['Duration'].astype(int)

#%%

handled['Duration'].mean()

#%%

handled.describe()

#%%

handled.columns

#%%

handled['language'] = pd.np.where(handled.Caller_ID.str.contains("EN:"), "english", pd.np.where(handled.Caller_ID.str.contains("ES:"),"spanish", pd.np.where(handled.Caller_ID.str.contains("FR:"), "french", "other")))

#%%

handled.head()

#%%

from datetime import datetime
#%%

handled['date'] = handled['date_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())

#%%

handled['day'] = handled['date'].apply(lambda x: datetime.strftime(x, '%A'))

#%%

aht_language = handled[['date', 'day', 'language', 'Duration']]

#%%

days = list(aht_language['day'].unique())

#%%

import matplotlib.pyplot as plt

for day in days:
	day_mask = aht_language['day'] == day
	print(day)

	c_aht_language = aht_language.loc[day_mask,:].groupby(['date', 'language'])['Duration'].mean().unstack().plot()

	c_count_language = aht_language.loc[day_mask,:].groupby(['date', 'language'])['Duration'].count().unstack().plot()

	plt.show()

	i = input("Continue?")

#%%

def get_program(caller_id):
	parts = caller_id.split(":")
	for p in parts:
		if len(p)>2:
			return p

	return "other"

#%%

handled['program'] = handled['Caller_ID'].apply(lambda x:get_program(x))

#%%

programs = list(handled['program'].unique())


#%%
import matplotlib.pyplot as plt
import datetime

for p in programs:
	program_mask = handled['program'] == p

	if (handled.loc[program_mask,'ID'].count()) > 100:

		print (p)

		c_aht_program = handled.loc[program_mask,:].groupby(['date', 'language'])['Duration'].count().unstack()

		columns = list(c_aht_program.columns)
		zscore = 0
		zscore_s = 0
		if 'english' in columns:
			calc_data = c_aht_program[:-1]
			zscore = ((c_aht_program.iloc[-1,:]['english']-calc_data['english'].mean())/calc_data['english'].std())

		if 'spanish' in columns:
			calc_data = c_aht_program[:-1]
			zscore_s = ((c_aht_program.iloc[-1,:]['spanish']-calc_data['spanish'].mean())/calc_data['spanish'].std())

		if zscore >= 1.65 or zscore_s >= 1.65:
			print(zscore, zscore_s)

			c_aht_program.plot(legend=True)


			plt.show()

			i = input("continue")

			#break

#%%
c_count_language = aht_language.groupby(['day', 'language'])['Duration'].count().unstack()

#%%

c_count_language.plot()

#%%

ticket_dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\phone_tickets_apr_8.csv")

#%%
ticket_dataset.head()

#%%

from datetime import datetime

ticket_dataset['date'] = ticket_dataset['create_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())

#%%

ticket_dataset['day'] = ticket_dataset['date'].apply(lambda x: datetime.strftime(x, '%A'))

#%%

by_program = ticket_dataset.groupby(['date', 'queue_id'])['id'].count().unstack()

#%%

by_program.head()

#%%
programs = list(by_program.columns)

#%%

for p in programs:
	c_data = by_program[:-1]
	if by_program[p].sum() > 100:
		zscore = 0
		zscore = ((by_program.iloc[-1,:][p]-c_data[p].mean())/c_data[p].std())

		if zscore >= 1.65:
			print (p, zscore)
			p_s = by_program[p]
			p_s.plot()
			plt.show()
			i = input("continue")

#%%

reasons_dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\reasons_contact.csv", index_col=0)



#%%

by_reason = ticket_dataset.groupby(['date', 'service_id'])['id'].count().unstack()

#%%
#remove last day
by_reason = by_reason[:-1]

#%%

reasons = list(by_reason.columns)

#%%

for r in reasons:
	c_data = by_reason[:-1]
	if by_reason[r].sum() > 100:
		zscore = 0
		zscore = ((by_reason.iloc[-1,:][r]-c_data[r].mean())/c_data[r].std())
		#print (reasons_dataset.loc[r], zscore)
		if zscore >= 1.65:
			print (reasons_dataset.loc[r], zscore)
		else:
			print (reasons_dataset.loc[r], zscore)
			p_s = by_reason[r]
			p_s.plot()
			plt.show()
			i = input("continue")

#%%


