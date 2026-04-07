if M._rep.domain not in (ZZ, QQ):
    if all(x.is_number for x in M) and M.has(Float):
        return _eigenvals_mpmath(M, multiple=multiple)  # Returns early!

if rational:  # This never gets reached when floats are present
    ...