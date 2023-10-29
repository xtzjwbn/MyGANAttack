def check_type(v, type_name, string):
	check_none(v, string)
	if not isinstance(v, type_name)	:
			raise ValueError(string + f" is given wrong type. {type_name} is needed.")

def check_none(v, string):
	if v is None:
		raise ValueError(string + " can not be None!")
