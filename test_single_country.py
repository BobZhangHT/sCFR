#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to analyze a single country"""

import real_data_analysis

# Test with a single country
if __name__ == "__main__":
    result = real_data_analysis.analyze_single_country(
        country_name="United States of America",
        start_date="2020-01-01",
        end_date="2021-12-31"
    )
    
    if result:
        print(f"\n{'='*80}")
        print("TEST RESULT SUMMARY")
        print(f"{'='*80}")
        print(f"Country: {result.country_name}")
        print(f"All Criteria Pass: {result.all_criteria_pass}")
        print(f"Score: {result.score}/4")
    else:
        print("Analysis failed")


