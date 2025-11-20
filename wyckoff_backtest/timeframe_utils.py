#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
timeframe_utils.py

Timeframe conversion utilities for backtest reporting
- Parse timeframe strings (e.g., "1H", "4H", "1D")
- Convert bar counts to actual time units
- Format durations for display
"""

from dataclasses import dataclass
from typing import Optional, Dict
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeframeInfo:
    """Information about a parsed timeframe"""
    value: int          # e.g., 1, 4
    unit: str           # 'H', 'D', 'W', 'M'
    hours: float        # equivalent hours
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'value': self.value,
            'unit': self.unit,
            'hours': self.hours
        }


@dataclass
class DurationInfo:
    """Information about a duration in various time units"""
    bars: int
    hours: float
    days: float
    display: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'bars': self.bars,
            'hours': self.hours,
            'days': self.days,
            'display': self.display
        }


class TimeframeUtils:
    """Utility class for timeframe conversions"""
    
    # Mapping of timeframe units to hours
    UNIT_TO_HOURS = {
        'H': 1.0,      # Hour
        'D': 24.0,     # Day
        'W': 168.0,    # Week (7 days)
        'M': 720.0,    # Month (30 days approximation)
    }
    
    @staticmethod
    def parse_timeframe(timeframe_str: str) -> Optional[TimeframeInfo]:
        """
        Parse timeframe string to components
        
        Args:
            timeframe_str: e.g., "1H", "4H", "1D", "1W", "1M"
            
        Returns:
            TimeframeInfo object or None if parsing fails
            
        Examples:
            >>> parse_timeframe("1H")
            TimeframeInfo(value=1, unit='H', hours=1.0)
            >>> parse_timeframe("4H")
            TimeframeInfo(value=4, unit='H', hours=4.0)
            >>> parse_timeframe("1D")
            TimeframeInfo(value=1, unit='D', hours=24.0)
        """
        if not timeframe_str:
            logger.warning("Empty timeframe string provided")
            return None
            
        # Parse pattern: number followed by unit letter
        pattern = r'^(\d+)([HDWM])$'
        match = re.match(pattern, timeframe_str.upper())
        
        if not match:
            logger.warning(f"Invalid timeframe format: {timeframe_str}")
            return None
            
        value = int(match.group(1))
        unit = match.group(2)
        
        if unit not in TimeframeUtils.UNIT_TO_HOURS:
            logger.warning(f"Unknown timeframe unit: {unit}")
            return None
            
        hours = value * TimeframeUtils.UNIT_TO_HOURS[unit]
        
        return TimeframeInfo(value=value, unit=unit, hours=hours)
    
    @staticmethod
    def bars_to_time(bars: int, timeframe_str: str) -> DurationInfo:
        """
        Convert bar count to time units
        
        Args:
            bars: Number of bars
            timeframe_str: e.g., "1H", "4H", "1D"
            
        Returns:
            DurationInfo object with converted time units
            
        Examples:
            >>> bars_to_time(954, "1H")
            DurationInfo(bars=954, hours=954.0, days=39.75, display="954 hours (39.75 days)")
            >>> bars_to_time(100, "4H")
            DurationInfo(bars=100, hours=400.0, days=16.67, display="400 hours (16.67 days)")
        """
        if bars < 0:
            logger.error(f"Negative bar count: {bars}")
            return DurationInfo(
                bars=bars,
                hours=0.0,
                days=0.0,
                display="Invalid duration"
            )
            
        if bars == 0:
            return DurationInfo(
                bars=0,
                hours=0.0,
                days=0.0,
                display="0 hours (0 days)"
            )
        
        # Parse timeframe
        tf_info = TimeframeUtils.parse_timeframe(timeframe_str)
        
        if tf_info is None:
            # Fallback: just show bars
            logger.info(f"Using fallback display for {bars} bars (invalid timeframe: {timeframe_str})")
            return DurationInfo(
                bars=bars,
                hours=0.0,
                days=0.0,
                display=f"{bars} bars"
            )
        
        # Calculate time units
        hours = bars * tf_info.hours
        days = hours / 24.0
        
        # Format display string
        display = TimeframeUtils._format_display(bars, hours, days, tf_info.unit)
        
        return DurationInfo(
            bars=bars,
            hours=hours,
            days=days,
            display=display
        )
    
    @staticmethod
    def _format_display(bars: int, hours: float, days: float, unit: str) -> str:
        """
        Format duration for display based on timeframe unit
        
        Args:
            bars: Number of bars
            hours: Total hours
            days: Total days
            unit: Timeframe unit ('H', 'D', 'W', 'M')
            
        Returns:
            Formatted string
        """
        if unit == 'H':
            # For hourly timeframes, show hours and days
            if hours < 24:
                return f"{int(hours)} hours"
            else:
                return f"{int(hours)} hours ({days:.2f} days)"
        elif unit == 'D':
            # For daily timeframes, just show days
            return f"{int(days)} days"
        elif unit == 'W':
            # For weekly timeframes, show weeks and days
            weeks = days / 7.0
            return f"{weeks:.2f} weeks ({int(days)} days)"
        elif unit == 'M':
            # For monthly timeframes, show months and days
            months = days / 30.0
            return f"{months:.2f} months ({int(days)} days)"
        else:
            # Fallback
            return f"{bars} bars"
    
    @staticmethod
    def format_duration(bars: int, timeframe_str: Optional[str]) -> str:
        """
        Format duration for display (convenience method)
        
        Args:
            bars: Number of bars
            timeframe_str: e.g., "1H", or None
            
        Returns:
            Formatted string
            
        Examples:
            >>> format_duration(954, "1H")
            "954 hours (39.75 days)"
            >>> format_duration(100, None)
            "100 bars"
        """
        if timeframe_str is None:
            logger.info(f"No timeframe provided, displaying {bars} bars")
            return f"{bars} bars"
            
        duration_info = TimeframeUtils.bars_to_time(bars, timeframe_str)
        return duration_info.display


if __name__ == "__main__":
    # Quick test
    print("Testing TimeframeUtils...")
    
    # Test parsing
    tf = TimeframeUtils.parse_timeframe("1H")
    print(f"1H parsed: {tf}")
    
    # Test conversion
    duration = TimeframeUtils.bars_to_time(954, "1H")
    print(f"954 bars @ 1H: {duration.display}")
    
    duration = TimeframeUtils.bars_to_time(100, "4H")
    print(f"100 bars @ 4H: {duration.display}")
    
    duration = TimeframeUtils.bars_to_time(30, "1D")
    print(f"30 bars @ 1D: {duration.display}")
    
    # Test format_duration
    print(f"Format 954 @ 1H: {TimeframeUtils.format_duration(954, '1H')}")
    print(f"Format 100 @ None: {TimeframeUtils.format_duration(100, None)}")
