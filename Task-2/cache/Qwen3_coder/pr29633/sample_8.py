if M._rep.domain not in (ZZ, QQ):
    # Skip this check for ZZ/QQ because it can be slow
    if all(x.is_number for x in M) and M.has(Float):
        return _eigenvals_mpmath(M, multiple=multiple)