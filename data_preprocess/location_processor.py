"""
地理位置处理模块
处理空地址、统一城市格式等
"""
import re
from typing import Optional, Dict


class LocationProcessor:
    """地理位置处理器"""
    
    # 深圳区域映射
    SHENZHEN_DISTRICTS = {
        '南山区', '福田区', '罗湖区', '龙华区', '龙岗区', 
        '宝安区', '盐田区', '坪山区', '光明区', '大鹏新区'
    }
    
    # 主要城市列表
    MAJOR_CITIES = {
        '深圳', '北京', '上海', '广州', '杭州', '成都', '南京',
        '武汉', '西安', '苏州', '天津', '重庆', '长沙', '郑州'
    }
    
    @staticmethod
    def extract_city(address: str, default_city: str = '深圳') -> str:
        """
        从地址中提取城市
        
        Args:
            address: 地址字符串
            default_city: 默认城市（当无法提取时）
            
        Returns:
            城市名称
        """
        if not address or not isinstance(address, str):
            return default_city
        
        # 检查是否包含主要城市名
        for city in LocationProcessor.MAJOR_CITIES:
            if city in address:
                return city
        
        # 检查深圳区域
        for district in LocationProcessor.SHENZHEN_DISTRICTS:
            if district in address:
                return '深圳'
        
        return default_city
    
    @staticmethod
    def extract_district(address: str) -> Optional[str]:
        """
        提取区域信息（针对深圳）
        
        Args:
            address: 地址字符串
            
        Returns:
            区域名称，如果无法提取则返回None
        """
        if not address or not isinstance(address, str):
            return None
        
        for district in LocationProcessor.SHENZHEN_DISTRICTS:
            if district in address:
                return district
        
        return None
    
    @staticmethod
    def is_valid_address(address: str) -> bool:
        """
        检查地址是否有效
        
        Args:
            address: 地址字符串
            
        Returns:
            是否有效
        """
        if not address or not isinstance(address, str):
            return False
        
        address = address.strip()
        
        # 空字符串
        if not address:
            return False
        
        # 太短（少于3个字符）
        if len(address) < 3:
            return False
        
        # 包含"未知"或"待定"等
        invalid_keywords = ['未知', '待定', 'N/A', 'n/a', 'null', 'None']
        for keyword in invalid_keywords:
            if keyword in address:
                return False
        
        return True
    
    @staticmethod
    def normalize_address(address: str, default_city: str = '深圳') -> Dict[str, str]:
        """
        标准化地址信息
        
        Args:
            address: 原始地址
            default_city: 默认城市
            
        Returns:
            {
                'city': 城市,
                'district': 区域（可选）,
                'full_address': 完整地址,
                'is_valid': 是否有效
            }
        """
        is_valid = LocationProcessor.is_valid_address(address)
        
        if not is_valid:
            return {
                'city': default_city,
                'district': None,
                'full_address': f"{default_city}（地址待补充）",
                'is_valid': False
            }
        
        city = LocationProcessor.extract_city(address, default_city)
        district = LocationProcessor.extract_district(address) if city == '深圳' else None
        
        return {
            'city': city,
            'district': district,
            'full_address': address,
            'is_valid': True
        }


def test_location_processor():
    """测试用例"""
    test_addresses = [
        "深圳南山区科技园",
        "深圳福田区深南大道",
        "",
        None,
        "未知",
        "北京朝阳区",
        "深圳龙华区",
    ]
    
    print("="*80)
    print("地理位置处理测试")
    print("="*80)
    
    for addr in test_addresses:
        result = LocationProcessor.normalize_address(addr)
        print(f"\n原始: {addr}")
        print(f"结果: {result}")


if __name__ == '__main__':
    test_location_processor()

